import os
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .backbones.vit_pytorch import vit_base_patch16_224_TransReID
# from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss, ContrastiveLoss
from loss.triplet_loss import TripletLoss
# pose-net
from model.pose_net import SimpleHRNet


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm1 = self.norm1.requires_grad_()
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm2 = self.norm2.requires_grad_()
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, prototype, global_feat,
                     pos, query_pos):
        q = k = self.with_pos_embed(prototype, query_pos)
        prototype_2 = self.self_attn(q, k, value=prototype)[0]
        prototype = prototype + self.dropout1(prototype_2)
        prototype = self.norm1(prototype)
        out_prototype = self.multihead_attn(query=self.with_pos_embed(prototype, query_pos),
                                   key=self.with_pos_embed(global_feat, pos),
                                   value=global_feat)[0]      
        prototype = prototype + self.dropout2(out_prototype)
        prototype = self.norm2(prototype)
        prototype = self.linear2(self.dropout(self.activation(self.linear1(prototype))))
        prototype = prototype + self.dropout3(prototype)
        prototype = self.norm3(prototype)
        return prototype
    
    def forward(self, prototype, global_feat, pos=None, query_pos=None):
        return self.forward_post(prototype, global_feat, pos, query_pos)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): #nn.GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, prototype, global_feat,
                pos = None, query_pos = None):
        output = prototype
        intermediate = []
        for layer in self.layers:
            output = layer(output, global_feat,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output

class build_skeleton_transformer(nn.Module):

    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_skeleton_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.in_planes = 768
        self.pose_dim = 2048

        # pose config
        self.pose = SimpleHRNet(48,
                                17, 
                                cfg.MODEL.POSE_WEIGHT,
                                model_name='HRNet',
                                resolution=(256, 128),
                                interpolation=cv2.INTER_CUBIC,
                                multiperson=False,
                                return_heatmaps=True,
                                return_bounding_boxes=False,
                                max_batch_size=32,
                                device=torch.device("cuda")
                                )

        # self.alphapose = SingleImageAlphaPose(alphapose_args, alphapose_cfg, device=torch.device("cuda"))
        
        self.skeleton_threshold = cfg.MODEL.SKT_THRES

        # decoderlayer config
        self.num_head = cfg.MODEL.NUM_HEAD
        self.dim_forward = 2048
        self.decoder_drop = cfg.MODEL.DECODER_DROP_RATE
        self.drop_first = cfg.MODEL.DROP_FIRST

        # decoder config
        self.decoder_numlayer = cfg.MODEL.NUM_DECODER_LAYER
        self.decoder_norm = nn.LayerNorm(self.in_planes)
        
        # query setting
        self.num_query = cfg.MODEL.QUERY_NUM
        self.query_embed = nn.Embedding(cfg.MODEL.QUERY_NUM, self.in_planes).weight

        # part view based decoder
        self.transformerdecoderlayer = TransformerDecoderLayer(self.in_planes, self.num_head, self.dim_forward, self.decoder_drop, "relu", self.drop_first)
        self.transformerdecoder = TransformerDecoder(self.transformerdecoderlayer, self.decoder_numlayer, self.decoder_norm)

        print('using Transformer_type: {} as a encoder'.format(cfg.MODEL.TRANSFORMER_TYPE))
        
        # visual context encoder 
        # Thanks the authors of TransReID https://github.com/heshuting555/TransReID.git 
        self.base_vit = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        
        print('Loading pretrained ImageNet model......from {}'.format(model_path))
        if pretrain_choice == 'imagenet':
            self.base_vit.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
 
        block = self.base_vit.blocks[-1]
        layer_norm = self.base_vit.norm
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_decoder = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_decoder.bias.requires_grad_(False)
        self.bottleneck_decoder.apply(weights_init_kaiming)     
        self.non_skt_decoder = nn.BatchNorm1d(self.in_planes)
        self.non_skt_decoder.bias.requires_grad_(False)
        self.non_skt_decoder.apply(weights_init_kaiming) 

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'triplet':
            self.classifier = TripletLoss(margin=cfg.SOLVER.COSINE_MARGIN, hard_factor=cfg.SOLVER.HARD_FACTOR)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        for i in range(17):
            exec('self.classifier_{} = nn.Linear(self.in_planes, self.num_classes, bias=False)'.format(i+1))
            exec('self.classifier_{}.apply(weights_init_classifier)'.format(i+1))

        for i in range(17):
            exec('self.bottleneck_{} = nn.BatchNorm1d(self.in_planes)'.format(i+1))
            exec('self.bottleneck_{}.bias.requires_grad_(False)'.format(i+1))
            exec('self.bottleneck_{}.apply(weights_init_kaiming)'.format(i+1))

        self.classifier_encoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_encoder.apply(weights_init_classifier)
        self.classifier_decoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_decoder.apply(weights_init_classifier)
        self.pose_decoder_linear = nn.Linear(self.pose_dim , self.in_planes)
        self.pose_avg = nn.AdaptiveAvgPool2d((1, self.in_planes))
        self.non_parts = nn.AdaptiveAvgPool2d((1, self.in_planes))
        self.decoder_global = nn.AdaptiveAvgPool2d((1, self.in_planes))

        for i in range(self.num_query):
            exec('self.classifier_decoder_{} = nn.Linear(self.in_planes, self.num_classes, bias=False)'.format(i+1))
            exec('self.classifier_decoder_{}.apply(weights_init_classifier)'.format(i+1))

        for i in range(self.num_query):
            exec('self.bottleneck_decoder_{} = nn.BatchNorm1d(self.in_planes)'.format(i+1))
            exec('self.bottleneck_decoder_{}.bias.requires_grad_(False)'.format(i+1))
            exec('self.bottleneck_decoder_{}.apply(weights_init_kaiming)'.format(i+1))

    def forward(self, x, label=None, cam_label= None, view_label=None): #ht optinal

        bs, c, h, w = x.shape # [batch, 3, 256, 128]

        # HRNet:
        heatmaps, joints = self.pose.predict(x)
        heatmaps = torch.from_numpy(heatmaps).cuda()    #[bs, 17, 64, 32]

        heatmaps = heatmaps.view(bs, heatmaps.shape[1], -1) # [bs, 17, 2048]

        ttt = heatmaps.cpu().numpy()
        skt_ft = np.zeros((heatmaps.shape[0], heatmaps.shape[1]), dtype=np.float32)

        for i, heatmap in enumerate(ttt):  #[64]
            for j, joint in enumerate(heatmap): #[17]

                if max(joint) < self.skeleton_threshold:
                    skt_ft[i][j] = 1    # Eq 4 in paper

        skt_ft = torch.from_numpy(skt_ft).cuda()    #[64, 17]

        pose_align_wt = self.pose_decoder_linear(heatmaps)  #[bs, 17, 768] FC

        heat_wt = self.pose_avg(heatmaps) #[bs, 1, 768]

        features = self.base_vit(x, cam_label=cam_label, view_label=view_label) # [64, 129, 768] ViT

        # Input of decoder 
        decoder_value = features * heat_wt
        decoder_value = decoder_value.permute(1,0,2)

        # strip 
        feature_length = features.size(1) - 1   #128
        patch_length = feature_length // self.num_query  #128 // 17
        token = features[:, 0:1]
        x = features[:, 1:]
    
        sim_feat = []
        # Encoder group features
        for i in range(16):
            exec('b{}_local = x[:, patch_length*{}:patch_length*{}]'.format(i+1, i, i+1))

            exec('b{}_local_feat = self.b2(torch.cat((token, b{}_local), dim=1))'.format(i+1, i+1))
            # exec('print(b{}_local_feat.shape)'.format(i+1))
            exec('local_feat_{} = b{}_local_feat[:, 0]'.format(i+1, i+1))

            exec('sim_feat.append(local_feat_{})'.format(i+1))

        b17_local = x[:, patch_length*16:]
        b17_local_feat = self.b2(torch.cat((token, b17_local), dim=1))
        local_feat_17 = b17_local_feat[:, 0]
        sim_feat.append(local_feat_17)

        # inference list
        inf_encoder = []
        # BN
        for i in range(17):
            exec('local_feat_{}_bn = self.bottleneck_{}(local_feat_{})'.format(i+1, i+1, i+1))
            exec('inf_encoder.append(local_feat_{}_bn/17)'.format(i+1))

        feat = features[:, 0].unsqueeze(1) * heat_wt + features[:, 0].unsqueeze(1)

        feat = feat.squeeze(1)

        # f_gb feature from encoder
        global_out_feat = self.bottleneck(feat) #[bs, 768]

        # part views
        query_embed = self.query_embed

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        prototype = torch.zeros_like(query_embed)
        
        # part-view based decoder
        out = self.transformerdecoder(prototype, decoder_value, query_pos=query_embed)

        # part view features
        last_out = out.permute(1,0,2)   # [bs, num_query, 768]

        sim_decoder = torch.stack(sim_feat, dim=1)  #[bs, 17, 768]

        #  PFA 
        sim_decoder = PFA(sim_decoder, pose_align_wt) #[bs 17 768]

        #  PVM
        decoder_feature, ind = PVM(sim_decoder, last_out) #[bs num_query 768]

        decoder_gb = self.decoder_global(decoder_feature).squeeze(1)   #[bs, 1, 768]

        # non skt parts 
        out_non_parts = []
        # skt parts 
        out_skt_parts = []

        decoder_skt_feature = []

        decoder_non_feature = []

        for i in range(bs):
            non_skt_feat_list = []
            per_skt_feat_list = []

            skt_feat = skt_ft[i]   #[17]
            # non_zero_skt = torch.nonzero(skt_feat).squeeze(1) #[num]

            skt_part = skt_feat.cpu().numpy()
            skt_ind = np.argwhere(skt_part==0).squeeze(1) #[17-num] numpy type

            for j in range(decoder_feature.shape[1]):

                # version 1 use original heatmap label
                # if skt_feat[skt_ind[i][j]] == 0: 
                #     non_feat = decoder_feature[i, j, :]
                #     non_skt_feat_list.append(non_feat)

                if skt_feat[ind[i][j]] == 1: # version 2 use PVM label
                    non_feat = decoder_feature[i, j, :]
                    non_skt_feat_list.append(non_feat)

                else:
                    skt_based_feat = decoder_feature[i, j, :] #[768]
                    per_skt_feat_list.append(skt_based_feat)
    

            if len(non_skt_feat_list) == 0:
                zero_feature = torch.zeros_like(decoder_gb[i])
                non_skt_feat_list.append(zero_feature)         #TODO:
            non_skt_single = torch.stack(non_skt_feat_list, dim=0).unsqueeze(0)  #[1, len(nonzero), 768]、
            
            decoder_non_feature.append(non_skt_single)
            non_skt_single = self.non_parts(non_skt_single) #[1, 1, 768]
            out_non_parts.append(non_skt_single) # [[1,1,768], [1,1,768], ....] bs length

            if len(per_skt_feat_list) == 0:
                per_skt_feat_list.append(decoder_gb[i])         #TODO:
            skt_single = torch.stack(per_skt_feat_list, dim=0).unsqueeze(0)     #[1, x, 768]

            decoder_skt_feature.append(skt_single)
            skt_single = self.non_parts(skt_single) #[1, 1, 768]
            out_skt_parts.append(skt_single)    # [[1,1,768], [1,1,768], ....] bs length


        last_non_parts = torch.cat(out_non_parts, dim=0)    #[bs, 1, 768]

        last_skt_parts = torch.cat(out_skt_parts, dim=0)    #[bs, 1, 768]

        # output high-confidence keypoint features
        decoder_out = self.bottleneck_decoder(last_skt_parts[:, 0]) #[bs, 768]

        # output non-skt-parts
        non_skt_parts = self.non_skt_decoder(last_non_parts[:, 0]) 

        # TODO:use last out or decoder out ?? 
        out_score = self.classifier_decoder(decoder_out)

        # Only high-confidence guided features are used to compute loss
        decoder_list = []

        # pad zeros for high-confidence guided features to self.num_query
        for i in decoder_skt_feature:
            if i.shape[1] < self.num_query:
                pad = torch.zeros((1,self.num_query-i.shape[1], self.in_planes)).to(i.device)
                pad_feat = torch.cat([i, pad], dim=1)  #[1, num_query, 768]
                decoder_list.append(pad_feat)
            else:
                decoder_list.append(i)


        decoder_lt = torch.cat(decoder_list, dim=0) # [64, self.num_query, 768]

        decoder_feature = decoder_lt


        # decoder parts features
        decoder_feat = [decoder_out]
        decoder_inf = []
        for i in range(self.num_query):
            exec('b{}_deocder_local_feat = decoder_feature[:, {}]'.format(i+1, i))
            exec('decoder_feat.append(b{}_deocder_local_feat)'.format(i+1))
            exec('decoder_inf.append(b{}_deocder_local_feat/self.num_query)'.format(i+1))

        # decoder BN
        for i in range(self.num_query):
            exec('decoder_local_feat_{}_bn = self.bottleneck_decoder_{}(b{}_deocder_local_feat)'.format(i+1, i+1, i+1))

        encoder_feat = [global_out_feat] + sim_feat 

        if self.training:
            # encoder parts
            cls_score = self.classifier_encoder(global_out_feat)

            encoder_score = [cls_score]

            for i in range(17):
                
                exec('cls_score_{} = self.classifier_{}(local_feat_{}_bn)'.format(i+1, i+1, i+1))
                exec('encoder_score.append(cls_score_{})'.format(i+1))

            decoder_score = [out_score]

            # decoder parts
            for i in range(self.num_query):

                exec('decoder_cls_score_{} = self.classifier_decoder_{}(decoder_local_feat_{}_bn)'.format(i+1, i+1, i+1))
                exec('decoder_score.append(decoder_cls_score_{})'.format(i+1))

            return encoder_score, encoder_feat ,decoder_score, decoder_feat, non_skt_parts

        else:
            # Inferece concat
            inf_feat = [global_out_feat] + inf_encoder + [decoder_out] + decoder_inf
            inf_features = torch.cat(inf_feat, dim=1)

            return inf_features

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if 'w1' in i.replace('module.', '') or 'w2' in i.replace('module.', ''):
            #     continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def PVM(matrix, matrix1):
    '''
    @matrix shape [bs, 17, 768]
    @matrix1 shape [bs, x, 768] 
    '''

    assert matrix.shape[0] == matrix1.shape[0], 'Wrong shape'
    assert matrix.shape[2] == matrix1.shape[2], 'Wrong dimension'

    batch_size = matrix.shape[0] #[bs, 17, 768]
    # skt_num = matrix.shape[1]
    final_sim = F.cosine_similarity(matrix.unsqueeze(2), matrix1.unsqueeze(1), dim=3) #[bs, 17, x] 

    _, ind = torch.max(final_sim, dim=2)    # ind.shape [bs, x]

    
    sim_match = []
    for i in range(batch_size):
        org_mat = matrix[i] #[17, C]
        sim_mat = matrix1[i] #[x, C]
        shuffle_mat = []

        for j in range(ind.shape[1]):
            new = org_mat[ind[i][j]] + sim_mat[j]  #[C]
            new = new.unsqueeze(0)
            shuffle_mat.append(new)

        bs_mat = torch.cat(shuffle_mat, dim=0)

        sim_match.append(bs_mat)
    
    final_feature = torch.stack(sim_match, dim=0)   #[bs, x, 768]?

    return final_feature, ind

def PFA(matrix, matrix1):
    '''
    @matrix shape [bs, 17, 768]
    @matrix1 shape [bs, 17, 768]

    '''
    assert matrix.shape[0] == matrix1.shape[0], 'Wrong shape'
    assert matrix.shape[1] == matrix1.shape[1], 'Wrong skt num'

    batch_size = matrix.shape[0] #[bs, 17, 768]

    # skt_num = matrix.shape[1]

    pose_weighted_feat = matrix * matrix1   #[bs, 17, 768]

    final_sim = F.cosine_similarity(matrix.unsqueeze(2), pose_weighted_feat.unsqueeze(1), dim=3) #[bs, 17, x] 

    _, ind = torch.max(final_sim, dim=2)

    sim_match = []
    for i in range(batch_size):
        org_mat = matrix[i] #[17, C]
        sim_mat = pose_weighted_feat[i] #[17, C]
        shuffle_mat = []

        for j in range(ind.shape[1]):
            new = org_mat[j] + sim_mat[ind[i][j]]  #[C]
            new = new.unsqueeze(0)
            shuffle_mat.append(new)

        bs_mat = torch.cat(shuffle_mat, dim=0)

        sim_match.append(bs_mat)
    
    alignment_feat = torch.stack(sim_match, dim=0)   #[bs, 17, 768]?

    return alignment_feat

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID
}



def make_pfd(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'skeleton_transformer':
        if cfg.MODEL.JPM:
            model = build_skeleton_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building skeleton_transformer with PFA and PVM module ===========')
    else:
        raise RuntimeError('Not Support this model!')

    return model

