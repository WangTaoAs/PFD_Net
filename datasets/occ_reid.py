# encoding: utf-8
"""
@author:  Tao Wang
@contact: taowang@stu.pku.edu.cn
"""

import sys
import os
import os.path as osp
import glob
import re
import warnings

from .bases import BaseImageDataset

# from torchreid.data.datasets import ImageDataset
# from torchreid.utils import read_image
import cv2
import numpy as np

class Occluded_REID(BaseImageDataset):

    dataset_dir = 'Occluded_REID'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Occluded_REID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'occ_reid_test')
        self.query_dir = osp.join(self.dataset_dir, 'occ_reid_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'occ_reid_test')

        ## for occ_reid, partial_reid, partial_iLids
        # self.occ_reid_query_dir = osp.join(self.dataset_dir, 'occ_reid_query')
        # self.occ_reid_gallery_dir = osp.join(self.dataset_dir, 'occ_reid_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        # occ_query = self._process_dir(self.occ_reid_query_dir, relabel=False)
        # occ_gallery = self._process_dir(self.occ_reid_gallery_dir , relabel=False)

        # self.load_pose = isinstance(self.transform, tuple)
        # if self.load_pose:
        #     if self.mode == 'query':
        #         self.pose_dir = osp.join(self.data_dir, 'occluded_body_pose')
        #     elif self.mode=='gallery':
        #         self.pose_dir = osp.join(self.data_dir, 'whole_body_pose')
        #     else:
        #         self.pose_dir=''
        if verbose:
            print("=> OCC_REID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    # def process_dir(self, dir_path, relabel=False, is_query=True):
    #     img_paths = glob.glob(osp.join(dir_path,'*','*.tif'))
    #     if is_query:
    #         camid = 0
    #     else:
    #         camid = 1
    #     pid_container = set()
    #     for img_path in img_paths:
    #         img_name = img_path.split('/')[-1]
    #         pid = int(img_name.split('_')[0])
    #         pid_container.add(pid)
    #     pid2label = {pid:label for label, pid in enumerate(pid_container)}

    #     data = []
    #     cam_container = set()
    #     for img_path in img_paths:
    #         img_name = img_path.split('/')[-1]
    #         pid = int(img_name.split('_')[0])
    #         # print(pid)
    #         camid = int((img_name.split('_')[-1]).split('.')[0])
    #         # print(camid)
    #         assert 1 <= camid <= 8
    #         camid -= 1  # index starts from 0
    #         if relabel:
    #             pid = pid2label[pid]
    #         data.append((img_path, self.pid_begin + pid, camid, 1))
    #         cam_container.add(camid)
    #     print(cam_container, 'cam_container')
    #     return data
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
