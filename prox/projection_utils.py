# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import cv2
import numpy as np
import json
import open3d as o3d

import dbw_utils as dbw


class Projection():
    def __init__(self, calib_dir):
        with open(osp.join(calib_dir, 'IR.json'), 'r') as f:
            self.depth_cam = json.load(f)
        with open(osp.join(calib_dir, 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)

    def row(self, A):
        return A.reshape((1, -1))

    def col(self, A):
        return A.reshape((-1, 1))

    def unproject_depth_image(self, depth_image, cam):
        # 1.screen -> camera
        # 512x424 -> 217088
        # column [0, ..., 511, ..., 0, ..., 511] (217088,)
        us = np.arange(depth_image.size) % depth_image.shape[1]
        # row [0, ..., 0, ..., 423, ..., 423] (217088,)
        vs = np.arange(depth_image.size) // depth_image.shape[1]
        # (217088,)
        ds = depth_image.ravel()
        # [[us], [vs], [ds]].T (217088,3)
        uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
        # (217088,1,2)
        xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                      np.asarray(cam['camera_mtx']),
                                                      np.asarray(cam['k']))
        # [[xy_undistorted_camspace], [ds]] (217088,3)
        xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), self.col(uvd[:, 2])))

        # 2.camera -> world
        # scale x,y by z
        xyz_camera_space[:, :2] *= self.col(xyz_camera_space[:, 2])
        # translate
        other_answer = xyz_camera_space - self.row(np.asarray(cam['view_mtx'])[:, 3])
        # rotate
        xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])

        return xyz.reshape((depth_image.shape[0], depth_image.shape[1], -1))

    def projectPoints(self, v, cam):
        v = v.reshape((-1, 3)).copy()
        return cv2.projectPoints(v, np.asarray(cam['R']),
                                 np.asarray(cam['T']),
                                 np.asarray(cam['camera_mtx']),
                                 np.asarray(cam['k']))[0].squeeze()

    def create_scan(self,
                    mask,
                    depth_im,
                    color_im=None,
                    mask_on_color=False,
                    coord='color',
                    TH=1e-2,
                    default_color=[1.00, 0.75, 0.80],
                    keypoints=None,
                    drop_noise=False,
                    eps=0.2,
                    min_samples=100,
                    new_depth_loss=False,
                    hp_fn=None):
        if not mask_on_color:
            depth_im[mask != 0] = 0

        if depth_im.size == 0:
            return {'v': []}

        # 1.compute point cloud of the image
        # screen -> world (depth camera)
        # (217088,3)
        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        # (217088,3)
        colors = np.tile(default_color, [points.shape[0], 1])

        # vis = o3d.Visualizer()
        # vis.create_window()
        # o3d_points = o3d.PointCloud()
        # o3d_points.points = o3d.Vector3dVector(points)
        # vis.add_geometry(o3d_points)
        # vis.run()

        # 2.compute point cloud of human body
        # world -> screen (color camera)
        # (217088,2)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)

        # (217088,)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        # (217088,)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        # (217088,)
        valid_idx = np.logical_and(valid_x, valid_y)
        if mask_on_color:
            # human body mask (217088,)
            valid_mask_idx = valid_idx.copy()
            valid_mask_idx[valid_mask_idx == True] = \
                mask[uvs[valid_idx == True][:, 1],
                     uvs[valid_idx == True][:, 0]] == 0
            # 2D human body (25831,2)
            uvs = uvs[valid_mask_idx == True]
            # human body point cloud (25831,3)
            points = points[valid_mask_idx]
            # (25831,3)
            colors = np.tile(default_color, [points.shape[0], 1])
            if color_im is not None:
                colors[:, :3] = color_im[uvs[:, 1], uvs[:, 0]] / 255.0
        else:
            # 2D human body (24696,2)
            uvs = uvs[valid_idx == True]
            # human body point cloud (24696,3)
            points = points[valid_idx == True]
            # (24696,3)
            colors = np.tile(default_color, [points.shape[0], 1])
            if color_im is not None:
                colors[:, :3] = color_im[uvs[:, 1], uvs[:, 0]] / 255.0

        labels = None
        if new_depth_loss:
            if keypoints is not None:
                labels = dbw.compute_points_label(uvs, keypoints, hp_fn)
            else:
                print("Keypoints is not available...")

        if coord == 'color':
            # world -> camera
            # (4,4)
            T = np.concatenate([np.asarray(self.color_cam['view_mtx']),
                                np.array([0, 0, 0, 1]).reshape(1, -1)])
            stacked = np.column_stack((points, np.ones(len(points))))
            # (25831,3)
            points = np.dot(T, stacked.T).T[:, :3]
            points = np.ascontiguousarray(points)

        if drop_noise:
            ind = dbw.drop_noise(points[points[:, 2] > TH], eps, min_samples)
        else:
            ind = points[:, 2] > TH

        return {'points': points[ind],
                'colors': colors[ind],
                'labels': None if labels is None else labels[ind]}

    def align_color2depth(self, depth_im, color_im, interpolate=True):
        (w_d, h_d) = (512, 424)

        if interpolate:
            # (217088,) [True, False, ...]
            zero_mask = np.array(depth_im == 0.).ravel()
            # (217088,)
            depth_im_flat = depth_im.ravel()
            depth_im_flat[zero_mask] = np.interp(np.flatnonzero(zero_mask),
                                                 np.flatnonzero(~zero_mask),
                                                 depth_im_flat[~zero_mask])
            depth_im = depth_im_flat.reshape(depth_im.shape)

        # screen -> camera (depth)
        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        # camera -> screen (color)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)

        valid_x = np.logical_and(uvs[:, 1] >= 0,
                                 uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0,
                                 uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]

        aligned_color = np.zeros((h_d, w_d, 3)).astype(color_im.dtype)
        aligned_color[valid_idx.reshape(h_d, w_d)] = color_im[uvs[:, 1], uvs[:, 0]]

        return aligned_color

    def align_depth2color(self, depth_im, depth_raw):
        (w_rgb, h_rgb) = (1920, 1080)
        (w_d, h_d) = (512, 424)

        # screen -> camera (depth)
        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)

        # camera -> screen (color)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)

        valid_x = np.logical_and(uvs[:, 1] >= 0,
                                 uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0,
                                 uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]

        aligned_depth = np.zeros((h_rgb, w_rgb)).astype('uint16')
        aligned_depth[uvs[:, 1], uvs[:, 0]] = depth_raw[valid_idx.reshape(h_d, w_d)]

        return aligned_depth
