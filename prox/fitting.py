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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from psbody.mesh.visibility import visibility_compute
from psbody.mesh import Mesh

import misc_utils as utils
import dist_chamfer as ext
distChamfer = ext.chamferDist()


import dbw_utils as dbw

@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               use_internal=True,
               **kwargs):
    ''' Initialize the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''
    if use_internal:
        body_pose = vposer.forward(pose_embedding).view(1, -1) \
            if use_vposer else None
    else:
        body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1) \
            if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose,
                   return_verts=False,
                   return_full_pose=False)

    # 1.z: est_d
    joints_3d = output.joints
    # joints_2d: gt_joints
    joints_2d = joints_2d.to(device=joints_3d.device)
    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])
    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)
    # sqrt(sum(square()))
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()
    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    # mean()
    height3d = length_3d.mean(dim=1)
    height2d = length_2d.mean(dim=1)
    est_d = focal_length * (height3d / height2d)

    # 2.x, y: 0
    # the human body is on the z axis for simplicity
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size],
                          device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()

    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)

    return init_t


class FittingMonitor(object):
    def __init__(self,
                 summary_steps=1,
                 visualize=False,
                 maxiters=100,
                 ftol=2e-09,
                 gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 viz_mode='mv',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type
        self.visualize = visualize
        self.viz_mode = viz_mode

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            if self.viz_mode == 'o3d':
                self.vis_o3d = o3d.Visualizer()
                self.vis_o3d.create_window()
                self.body_o3d = o3d.TriangleMesh()
                self.scan = o3d.PointCloud()
            else:
                self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            if self.viz_mode == 'o3d':
                self.vis_o3d.close()
            else:
                self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    # optimize
    def run_fitting(self,
                    optimizer,
                    closure,
                    params,
                    body_model,
                    use_vposer=True,
                    pose_embedding=None,
                    vposer=None,
                    **kwargs):
        '''
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                    The final loss value
        '''
        append_wrists = \
            self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            # 1.forward (closure -> 1, 2)
            # 2.backward (closure -> 3)
            # 3.update parameters (optimizer)
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())
                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            prev_loss = loss.item()

        return prev_loss

    # create a closure
    def create_fitting_closure(self,
                               optimizer,
                               body_model,
                               camera=None,
                               gt_joints=None,
                               loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True,
                               return_full_pose=False,
                               use_vposer=False,
                               vposer=None,
                               pose_embedding=None,
                               scan_tensor=None,
                               create_graph=False,
                               use_internal=True,
                               gt_joints_3d=None,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = \
            self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            if use_internal:
                body_pose = vposer.forward(pose_embedding).view(1, -1) \
                    if use_vposer else None
            else:
                body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1) \
                    if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            # 1.body model: forward
            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)

            # 2.loss: forward
            total_loss = loss(body_model_output,
                              camera=camera,
                              gt_joints=gt_joints,
                              body_model=body_model,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              scan_tensor=scan_tensor,
                              visualize=self.visualize,
                              gt_joints_3d=gt_joints_3d,
                              **kwargs)

            # 3.loss: backward
            if backward:
                total_loss.backward(create_graph=create_graph)

            # 4.visualize
            if self.visualize:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                if self.steps == 0 and self.viz_mode == 'o3d':
                    self.body_o3d.vertices = o3d.Vector3dVector(vertices.squeeze())
                    self.body_o3d.triangles = o3d.Vector3iVector(body_model.faces)
                    self.body_o3d.vertex_normals = o3d.Vector3dVector([])
                    self.body_o3d.triangle_normals = o3d.Vector3dVector([])
                    self.body_o3d.compute_vertex_normals()
                    self.vis_o3d.add_geometry(self.body_o3d)

                    if scan_tensor is not None:
                        self.scan.points = o3d.Vector3dVector(scan_tensor.detach().cpu().numpy().squeeze())
                        N = np.asarray(self.scan.points).shape[0]
                        self.scan.colors = o3d.Vector3dVector(np.tile([1.00, 0.75, 0.80], [N, 1]))
                        self.vis_o3d.add_geometry(self.scan)

                    self.vis_o3d.update_geometry()
                    self.vis_o3d.poll_events()
                    self.vis_o3d.update_renderer()
                elif self.steps % self.summary_steps == 0:
                    if self.viz_mode == 'o3d':
                        self.body_o3d.vertices = o3d.Vector3dVector(vertices.squeeze())
                        self.body_o3d.triangles = o3d.Vector3iVector(body_model.faces)
                        self.body_o3d.vertex_normals = o3d.Vector3dVector([])
                        self.body_o3d.triangle_normals = o3d.Vector3dVector([])
                        self.body_o3d.compute_vertex_normals()

                        self.vis_o3d.update_geometry()
                        self.vis_o3d.poll_events()
                        self.vis_o3d.update_renderer()
                    else:
                        self.mv.update_mesh(vertices.squeeze(),
                                            body_model.faces)

            self.steps += 1

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):
    def __init__(self,
                 search_tree=None,
                 pen_distance=None,
                 tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None,
                 right_hand_prior=None,
                 interpenetration=True,
                 dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0,
                 jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 s2m=False,
                 m2s=False,
                 rho_s2m=1,
                 rho_m2s=1,
                 s2m_weight=0.0,
                 m2s_weight=0.0,
                 head_mask=None,
                 body_mask=None,
                 sdf_penetration=False,
                 voxel_size=None,
                 grid_min=None,
                 grid_max=None,
                 sdf=None,
                 sdf_normals=None,
                 sdf_penetration_weight=0.0,
                 R=None,
                 t=None,
                 contact=False,
                 contact_loss_weight=0.0,
                 contact_verts_ids=None,
                 rho_contact=0.0,
                 contact_angle=0.0,
                 new_depth_loss=False,
                 scan_labels=None,
                 vertices_labels=None,
                 use_joint_loss_3d=False,
                 joints_3d_weight=1.0,
                 joint_penetration=False,
                 joint_penetration_weight=1.0,
                 **kwargs):
        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.s2m = s2m
        self.m2s = m2s
        self.s2m_robustifier = utils.GMoF(rho=rho_s2m)
        self.m2s_robustifier = utils.GMoF(rho=rho_m2s)

        self.body_pose_prior = body_pose_prior
        self.shape_prior = shape_prior

        self.body_mask = body_mask
        self.head_mask = head_mask

        self.R = R
        self.t = t

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))
        self.register_buffer('s2m_weight',
                             torch.tensor(s2m_weight, dtype=dtype))
        self.register_buffer('m2s_weight',
                             torch.tensor(m2s_weight, dtype=dtype))

        self.sdf_penetration = sdf_penetration
        self.joint_penetration = joint_penetration
        if self.sdf_penetration or self.joint_penetration:
            self.sdf = sdf
            self.sdf_normals = sdf_normals
            self.voxel_size = voxel_size
            self.grid_min = grid_min
            self.grid_max = grid_max
            self.register_buffer('sdf_penetration_weight',
                                 torch.tensor(sdf_penetration_weight, dtype=dtype))
            self.register_buffer('joint_penetration_weight',
                                 torch.tensor(joint_penetration_weight, dtype=dtype))

        self.contact = contact
        if self.contact:
            self.contact_verts_ids = contact_verts_ids
            self.rho_contact = rho_contact
            self.contact_angle = contact_angle
            self.register_buffer('contact_loss_weight',
                                 torch.tensor(contact_loss_weight, dtype=dtype))
            self.contact_robustifier = utils.GMoF_unscaled(rho=self.rho_contact)

        self.new_depth_loss = new_depth_loss
        if new_depth_loss:
            print("Using new depth loss...")
        self.scan_labels = scan_labels
        self.vertices_labels = vertices_labels

        self.use_joint_loss_3d = use_joint_loss_3d
        self.register_buffer('joints_3d_weight',
                             torch.tensor(joints_3d_weight, dtype=dtype))

        self.cnt = 0

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self,
                body_model_output,
                camera,
                gt_joints,
                joints_conf,
                body_model_faces,
                joint_weights,
                use_vposer=False,
                pose_embedding=None,
                scan_tensor=None,
                visualize=False,
                scene_v=None,
                scene_vn=None,
                scene_f=None,
                ftov=None,
                gt_joints_3d=None,
                **kwargs):
        # 1.joint loss
        # (1) calculate the projected joints
        # camera -> screen
        projected_joints = camera(body_model_output.joints)
        # (2) calculate the weight for each joint
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf
                   else joint_weights).unsqueeze(dim=-1)
        # (3) calculate the distance of the projected joints from the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = torch.sum(weights ** 2 * joint_diff) * \
                     self.data_weight ** 2

        joint_loss_3d = 0.0
        if self.use_joint_loss_3d:
            for i in range(25):
                if gt_joints_3d[i] is None:
                    continue
                diff_norm = torch.norm(gt_joints_3d[i] - body_model_output.joints[0][i])
                if diff_norm > 0.05:
                    joint_loss_3d += joints_conf[0][i] * (diff_norm ** 2)
            joint_loss_3d = joint_loss_3d * self.joints_3d_weight

        # 2.body pose prior loss
        if use_vposer:
            pprior_loss = pose_embedding.pow(2).sum() * \
                          self.body_pose_weight ** 2
        else:
            # the second parameter of 'body_model_output.betas' is unnecessary
            pprior_loss = torch.sum(self.body_pose_prior(body_model_output.body_pose, body_model_output.betas)) *\
                          self.body_pose_weight ** 2

        # 3.shape loss
        shape_loss = torch.sum(self.shape_prior(body_model_output.betas)) * \
                     self.shape_weight ** 2

        # 4.angle prior loss
        # calculate the prior over the joint rotations
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(self.angle_prior(body_pose)) * \
                           self.bending_prior_weight ** 2

        # 5.hand pose prior loss
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(self.left_hand_prior(body_model_output.left_hand_pose)) * \
                                   self.hand_prior_weight ** 2
        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(self.right_hand_prior(body_model_output.right_hand_pose)) * \
                                    self.hand_prior_weight ** 2

        # 6.expression and jaw prior loss
        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(body_model_output.expression)) * \
                              self.expr_prior_weight ** 2
            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(self.jaw_prior(body_model_output.jaw_pose.mul(self.jaw_prior_weight)))

        # 7.interpenetration loss
        pen_loss = 0.0
        if self.interpenetration and self.coll_loss_weight.item() > 0:
            # (1) compute human body's triangles
            batch_size = projected_joints.shape[0]
            # size: BxFx3x3 (F represents the triangles number)
            triangles = torch.index_select(body_model_output.vertices, 1, body_model_faces)\
                .view(batch_size, -1, 3, 3)

            # (2) search for self intersection (collision index)
            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # (3) remove collision pairs
            # tri_filtering_module = filter_faces
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            # (4) compute interpenetration loss
            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(self.coll_loss_weight * self.pen_distance(triangles, collision_idxs))

        # 8.depth loss
        s2m_dist = 0.0
        m2s_dist = 0.0
        # calculate the scan2mesh and mesh2scan loss from the sparse point cloud
        if (self.s2m or self.m2s) and \
                (self.s2m_weight > 0 or self.m2s_weight > 0) and \
                scan_tensor is not None:
            # (1) create human body mesh
            # (1,N,3)
            vertices_np = body_model_output.vertices.detach().cpu().numpy().squeeze()
            # (M,3)
            body_faces_np = body_model_faces.detach().cpu().numpy().reshape(-1, 3)
            m = Mesh(v=vertices_np, f=body_faces_np)

            # (2) compute visibility of human body vertices
            vis, n_dot = visibility_compute(v=m.v,
                                            f=m.f,
                                            cams=np.array([[0.0, 0.0, 0.0]]))
            vis = vis.squeeze()

            if self.new_depth_loss:
                # [1] Calculate labels
                scan_labels_np = np.array(self.scan_labels)
                vertices_labels_np = np.array(self.vertices_labels)

                # [2] Transform vertices labels from 'smplx' to 'openpose'
                # smplx2openpose = utils.smpl_to_openpose(model_type='smplx',
                #                                         use_hands=True,
                #                                         use_face=True,
                #                                         use_face_contour=False,
                #                                         openpose_format='coco25')
                # for i in range(vertices_labels_np.shape[0]):
                #     if vertices_labels_np[i] in smplx2openpose:
                #         vertices_labels_np[i] = np.where(smplx2openpose == vertices_labels_np[i])[0][0]
                #     else:
                #         vertices_labels_np[i] = -2

                # [3] Unify labels
                common_labels = np.intersect1d(np.unique(scan_labels_np),
                                               np.unique(vertices_labels_np))
                for i in range(scan_labels_np.shape[0]):
                    if scan_labels_np[i] not in common_labels:
                        scan_labels_np[i] = 0
                for i in range(vertices_labels_np.shape[0]):
                    if vertices_labels_np[i] not in common_labels:
                        vertices_labels_np[i] = 0
                common_labels = np.append(common_labels, 0)
                for i in range(scan_labels_np.shape[0]):
                    scan_labels_np[i] = np.where(common_labels == scan_labels_np[i])[0][0]
                for i in range(vertices_labels_np.shape[0]):
                    vertices_labels_np[i] = np.where(common_labels == vertices_labels_np[i])[0][0]

                # [4] Visualize
                # 0: no visualization
                # 1: part by part
                # 2: step by step
                visualize_mode = 0
                scan_points = scan_tensor.cpu().detach().numpy()[0]
                body_points = body_model_output.vertices.cpu().detach().numpy()[0]
                if visualize_mode == 1:
                    print('-' * 10, 'Begin Debug', '-' * 10)
                    for i in range(common_labels.shape[0]):
                        print('Part {}'.format(common_labels[i]))
                        labels_color = dbw.gen_random_colors(common_labels.shape[0],
                                                             visualize_mode, i)
                        dbw.visualize_points_with_labels(scan_points,
                                                         scan_labels_np,
                                                         labels_color)
                        dbw.visualize_points_with_labels(body_points,
                                                         vertices_labels_np,
                                                         labels_color)
                    print('-' * 10, 'End Debug', '-' * 10)
                elif visualize_mode == 2:
                    labels_color = dbw.gen_random_colors(common_labels.shape[0])
                    dbw.visualize_points_with_labels(scan_points,
                                                     scan_labels_np,
                                                     labels_color)
                    dbw.visualize_points_with_labels(body_points,
                                                     vertices_labels_np,
                                                     labels_color)

                # [5] Calculate new depth loss
                if self.s2m and self.s2m_weight > 0 and vis.sum() > 0:
                    s2m_dist_list = []
                    for i in range(common_labels.shape[0]):
                        points_inds = np.where(scan_labels_np == common_labels[i])[0]
                        vertices_inds = np.where(np.logical_and(vis > 0,
                                                                vertices_labels_np == common_labels[i]))[0]
                        if points_inds.shape[0] == 0 or vertices_inds.shape[0] == 0:
                            continue
                        part_s2m_dist, _, _, _ = distChamfer(scan_tensor[:, points_inds, :],
                                                             body_model_output.vertices[:, vertices_inds, :])
                        s2m_dist_list.append(part_s2m_dist)
                    if len(s2m_dist_list) > 0:
                        s2m_dist = torch.cat(s2m_dist_list, dim=1)
                        s2m_dist = self.s2m_robustifier(s2m_dist.sqrt())
                        s2m_dist = self.s2m_weight * s2m_dist.sum()

                if self.m2s and self.m2s_weight > 0 and vis.sum() > 0:
                    m2s_dist_list = []
                    for i in range(common_labels.shape[0]):
                        points_inds = np.where(scan_labels_np == common_labels[i])[0]
                        vertices_inds = np.intersect1d(np.where(np.logical_and(vis > 0, self.body_mask))[0],
                                                       np.where(vertices_labels_np == common_labels[i])[0])
                        if points_inds.shape[0] == 0 or vertices_inds.shape[0] == 0:
                            continue
                        _, part_m2s_dist, _, _ = distChamfer(scan_tensor[:, points_inds, :],
                                                             body_model_output.vertices[:, vertices_inds, :])
                        m2s_dist_list.append(part_m2s_dist)
                    if len(m2s_dist_list) > 0:
                        m2s_dist = torch.cat(m2s_dist_list, dim=1)
                        m2s_dist = self.m2s_robustifier(m2s_dist.sqrt())
                        m2s_dist = self.m2s_weight * m2s_dist.sum()
            else:
                # (3) compute distance from scan to mesh
                if self.s2m and self.s2m_weight > 0 and vis.sum() > 0:
                    s2m_dist, _, _, _ = distChamfer(scan_tensor,
                                                    body_model_output.vertices[:, np.where(vis > 0)[0], :])
                    s2m_dist = self.s2m_robustifier(s2m_dist.sqrt())
                    s2m_dist = self.s2m_weight * s2m_dist.sum()

                # (4) compute distance from mesh to scan
                if self.m2s and self.m2s_weight > 0 and vis.sum() > 0:
                    _, m2s_dist, _, _ = distChamfer(scan_tensor,
                                                    body_model_output.vertices[:, np.where(
                                                        np.logical_and(vis > 0, self.body_mask))[0], :])
                    m2s_dist = self.m2s_robustifier(m2s_dist.sqrt())
                    m2s_dist = self.m2s_weight * m2s_dist.sum()

        # human body vertices: camera -> world
        if self.R is not None and self.t is not None:
            vertices = body_model_output.vertices
            nv = vertices.shape[1]
            vertices.squeeze_()
            vertices = self.R.mm(vertices.t()).t() +\
                       self.t.repeat([nv, 1])
            vertices.unsqueeze_(0)

            joints = body_model_output.joints
            nj = joints.shape[1]
            joints = joints.squeeze_()
            joints = self.R.mm(joints.t()).t() + self.t.repeat([nj, 1])
            joints.unsqueeze_(0)

        # 9.sdf penetration loss
        sdf_penetration_loss = 0.0
        if self.sdf_penetration and self.sdf_penetration_weight > 0:
            # (1) compute human body vertice's indexes
            grid_dim = self.sdf.shape[0]
            sdf_ids = torch.round((vertices.squeeze() - self.grid_min) / self.voxel_size).\
                to(dtype=torch.long)
            sdf_ids.clamp_(min=0, max=grid_dim-1)

            # (2) normalize human body vertices
            norm_vertices = (vertices - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1

            # (3) compute human body vertice's sdf normals
            body_sdf = F.grid_sample(self.sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                     norm_vertices[:, :, [2, 1, 0]].view(1, nv, 1, 1, 3),
                                     padding_mode='border')
            if self.sdf_normals is not None:
                sdf_normals = self.sdf_normals[sdf_ids[:, 0], sdf_ids[:, 1], sdf_ids[:, 2]]
            else:
                sdf_normals = self.sdf_normals

            # (4) compute sdf penetration loss
            if body_sdf.lt(0).sum().item() < 1:
                sdf_penetration_loss = torch.tensor(0.0, dtype=joint_loss.dtype, device=joint_loss.device)
            else:
                if sdf_normals is None:
                    sdf_penetration_loss = self.sdf_penetration_weight * \
                                           (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs())\
                                               .pow(2).sum(dim=-1).sqrt().sum()
                else:
                    sdf_penetration_loss = self.sdf_penetration_weight * \
                                           (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() *
                                            sdf_normals[body_sdf.view(-1) < 0, :]).\
                                               pow(2).sum(dim=-1).sqrt().sum()

        # 10.contact loss
        contact_loss = 0.0
        if self.contact and self.contact_loss_weight > 0:
            # (1) get vertices that are possible to contact with environment (from a file)
            contact_body_vertices = vertices[:, self.contact_verts_ids, :]

            # (2) calculate contact distance and contact vertices index
            # scene_v: vertices of the scene
            contact_dist, _, idx1, _ = distChamfer(contact_body_vertices.contiguous(),
                                                   scene_v)

            # (3) calculate all triangular face normals and vertices normals (human body)
            # a.calculate triangular faces
            body_triangles = torch.index_select(vertices, 1, body_model_faces).\
                view(1, -1, 3, 3)
            # b.calculate edges of the triangles
            edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
            edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
            # c.calculate triangular face normals using cross product
            body_normals = torch.cross(edge0, edge1, dim=2)
            # normalize the result to get a unit vector
            body_normals = body_normals / torch.norm(body_normals, 2, dim=2, keepdim=True)
            # d.compute vertices normals and normalize them too
            body_v_normals = torch.mm(ftov, body_normals.squeeze())
            body_v_normals = body_v_normals / torch.norm(body_v_normals, 2, dim=1, keepdim=True)

            # (4) select contact vertices normals
            contact_body_verts_normals = body_v_normals[self.contact_verts_ids, :]

            # (5) select contact scene vertices normals
            contact_scene_normals = scene_vn[:, idx1.squeeze().to(dtype=torch.long), :].squeeze()

            # (6) calculate the angle between contact vertices normals and contact scene vertices normals
            angles = torch.asin(
                torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=1, keepdim=True)) \
                     *180 / np.pi

            # (7) filter contact vertices that do not match the condition
            valid_contact_mask = (angles.le(self.contact_angle) + angles.ge(180 - self.contact_angle)).ge(1)
            valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()

            # (8) compute contact loss
            contact_dist = self.contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())
            contact_loss = self.contact_loss_weight * contact_dist.mean()

        # 11.joint penetration loss
        joint_penetration_loss = 0.0
        if self.joint_penetration:
            # (1) compute human body joint's indexes
            grid_dim = self.sdf.shape[0]
            sdf_ids = torch.round((joints.squeeze() - self.grid_min) / self.voxel_size). \
                to(dtype=torch.long)
            sdf_ids.clamp_(min=0, max=grid_dim - 1)

            # (2) normalize human body joints
            norm_joints = (joints - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1

            # (3) compute human body joint's sdf normals
            body_sdf = F.grid_sample(self.sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                     norm_joints[:, :, [2, 1, 0]].view(1, nj, 1, 1, 3),
                                     padding_mode='border')
            if self.sdf_normals is not None:
                sdf_normals = self.sdf_normals[sdf_ids[:, 0], sdf_ids[:, 1], sdf_ids[:, 2]]
            else:
                sdf_normals = self.sdf_normals

            # (4) compute sdf penetration loss
            if body_sdf.lt(0).sum().item() < 1:
                joint_penetration_loss = torch.tensor(0.0, dtype=joint_loss.dtype, device=joint_loss.device)
            else:
                if sdf_normals is None:
                    joint_penetration_loss = self.joint_penetration_weight * \
                                             (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs()).\
                                                 pow(2).sum(dim=-1).sqrt().sum()
                else:
                    joint_penetration_loss = self.joint_penetration_weight * \
                                             (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() *
                                              sdf_normals[body_sdf.view(-1) < 0, :]).\
                                                 pow(2).sum(dim=-1).sqrt().sum()

        total_loss = (joint_loss +
                      pprior_loss +
                      shape_loss +
                      angle_prior_loss +
                      pen_loss +
                      jaw_prior_loss +
                      expression_loss +
                      left_hand_prior_loss +
                      right_hand_prior_loss +
                      s2m_dist +
                      m2s_dist +
                      sdf_penetration_loss +
                      contact_loss +
                      joint_penetration_loss)
        if self.use_joint_loss_3d:
            total_loss += joint_loss_3d

        vis_body = False
        if self.cnt % 100 == 0:
            print('*' * 30)
            print('total:{:.2f}'.format(total_loss.item()))
            print('joint_loss:{:.2f}'.format(joint_loss.item()))
            if self.use_joint_loss_3d:
                print('joint_loss_3d:{:.2f}'.format(joint_loss_3d.item()))
            print('pprior_loss:{:.2f}'.format(pprior_loss.item()))
            print('shape_loss:{:.2f}'.format(shape_loss.item()))
            print('angle_prior_loss:{:.2f}'.format(angle_prior_loss.item()))
            print('pen_loss:{:.2f}'.format(pen_loss))
            print('jaw_prior_loss:{:.2f}'.format(jaw_prior_loss))
            print('expression_loss:{:.2f}'.format(expression_loss))
            print('left_hand_prior_loss:{:.2f}'.format(left_hand_prior_loss.item()))
            print('right_hand_prior_loss:{:.2f}'.format(right_hand_prior_loss.item()))
            print('s2m_dist:{:.2f}'.format(s2m_dist))
            print('m2s_dist:{:.2f}'.format(m2s_dist))
            print('sdf_penetration_loss:{:.2f}'.format(sdf_penetration_loss))
            print('contact_loss:{:.2f}'.format(contact_loss))
            print('joint_penetration_loss:{:.2f}'.format(joint_penetration_loss))
            print('*' * 30, '\n')

            if vis_body:
                body_points = body_model_output.vertices.cpu().detach().numpy()
                vis = o3d.Visualizer()
                vis.create_window()
                o3d_points = o3d.PointCloud()
                o3d_points.points = o3d.Vector3dVector(body_points)
                vis.add_geometry(o3d_points)
                vis.run()
        self.cnt += 1

        return total_loss


class SMPLifyCameraInitLoss(nn.Module):
    def __init__(self,
                 init_joints_idxs,
                 trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2,
                 camera_mode='moving',
                 dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype
        self.camera_mode = camera_mode

        if trans_estimation is not None:
            self.register_buffer('trans_estimation',
                                 utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('init_joints_idxs',
                             utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, body_model, **kwargs):
        # 1.joint loss
        # camera -> screen
        projected_joints = camera(body_model_output.joints)
        joint_error = torch.pow(torch.index_select(gt_joints, 1, self.init_joints_idxs) -
                                torch.index_select(projected_joints, 1, self.init_joints_idxs), 2)
        joint_loss = torch.sum(joint_error) * \
                     self.data_weight ** 2

        # 2.depth loss
        depth_loss = 0.0
        if self.depth_loss_weight.item() > 0 and self.trans_estimation is not None:
            if self.camera_mode == 'moving':
                depth_loss = self.depth_loss_weight ** 2 * \
                             torch.sum((camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))
            elif self.camera_mode == 'fixed':
                depth_loss = self.depth_loss_weight ** 2 * \
                             torch.sum((body_model.transl[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
