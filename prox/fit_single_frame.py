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

import time
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import os.path as osp
import numpy as np
import torch
import torchgeometry as tgm
from tqdm import tqdm
from collections import defaultdict
import cv2
import PIL.Image as pil_img
import json
import scipy.sparse as sparse

from human_body_prior.tools.model_loader import load_vposer
from psbody.mesh import Mesh

import fitting
from optimizers import optim_factory
from vposer import VPoserEncoder, VPoserDecoder
import dbw_utils as dbw


def fit_single_frame(img,
                     keypoints,
                     init_trans,
                     scan,
                     scene_name,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     body_scene_rendering_fn='body_scene.png',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length_x=5000.,
                     focal_length_y=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     render_results=True,
                     camera_mode='moving',
                     s2m=False,
                     s2m_weights=None,
                     m2s=False,
                     m2s_weights=None,
                     rho_s2m=1,
                     rho_m2s=1,
                     init_mode=None,
                     trans_opt_stages=None,
                     viz_mode='mv',
                     sdf_penetration=False,
                     sdf_penetration_weights=0.0,
                     sdf_dir=None,
                     cam2world_dir=None,
                     contact=False,
                     rho_contact=1.0,
                     contact_loss_weights=None,
                     contact_angle=15,
                     contact_body_parts=None,
                     body_segments_dir=None,
                     load_scene=False,
                     scene_dir=None,
                     use_internal=True,
                     joints_3d_weight=None,
                     joint_penetration=False,
                     joint_penetration_weight=None,
                     **kwargs):
    # ****************************************
    # 1.check and initialize params (mainly all kinds of weights)
    # (1) there are 7 stages, so the dimension of all the weights should be 7
    # (2) original code has 5 stages, so the dimension of all default weights is 5
    # ****************************************
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    body_model.reset_params()
    body_model.transl.requires_grad = True

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # if visualize:
    #     pil_img.fromarray((img * 255).astype(np.uint8)).show()

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]
    msg = ('Number of Body pose prior weights {}'.format(len(body_pose_prior_weights)) +
           ' does not match the number of data term weights {}'.format(len(data_weights)))
    assert (len(data_weights) == len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) == len(body_pose_prior_weights)), msg

        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand joint distance weights')
        assert (len(hand_joints_weights) == len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) == len(body_pose_prior_weights)), \
        msg.format(len(shape_weights), len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) == len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) == len(body_pose_prior_weights)), \
            msg.format(len(body_pose_prior_weights), len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) == len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) == len(body_pose_prior_weights)), msg

    # ****************************************
    # 2.vposer related (vposer directory)
    # ****************************************
    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None] * 2

    if use_vposer:
        pose_embedding = torch.zeros([batch_size, vposer_latent_dim],
                                     device=device,
                                     requires_grad=True,
                                     dtype=dtype)

    if use_vposer:
        if use_internal:
            print("Use internal vposer...")
            vposer = VPoserDecoder(vposer_ckpt=vposer_ckpt,
                                   latent_dim=vposer_latent_dim,
                                   dtype=dtype,
                                   **kwargs)
        else:
            vposer_ckpt = osp.expandvars(vposer_ckpt)
            vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    use_ini_pose = kwargs.get('use_ini_pose')
    if use_ini_pose:
        ini_pose = dbw.get_ini_pose(keypoints,
                                    batch_size,
                                    23 if use_vposer else 21,
                                    dtype)
    if use_vposer:
        if use_internal:
            latent_mean = torch.zeros([batch_size, vposer_latent_dim],
                                      device=device,
                                      requires_grad=True,
                                      dtype=dtype)
            body_mean_pose = vposer(latent_mean).detach().cpu()
        else:
            body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                          dtype=dtype)
    else:
        if use_ini_pose:
            body_mean_pose = ini_pose.detach().cpu()
        else:
            body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    # ****************************************
    # 3.load the keypoints and point cloud
    # ****************************************
    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    # joints position: (1,N,2) (_,(_,(x,y)))
    gt_joints = keypoint_data[:, :, :2]
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        # joints confidence: (1,N)
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    scan_tensor = None
    if scan is not None:
        # scan: scan_dict
        # scan_tensor: scan_dict['points']
        scan_tensor = torch.tensor(scan.get('points'), device=device, dtype=dtype).unsqueeze(0)

    scan_labels = None
    vertices_labels = None
    if kwargs.get('new_depth_loss', False):
        scan_labels = scan.get('labels')
        my_vertices_np = body_model(return_verts=True,
                                     body_pose=torch.zeros((batch_size, 63), dtype=dtype, device=device))\
            .vertices.detach().cpu().numpy().squeeze()
        my_faces_np = body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3)
        part_segm_fn = os.path.expandvars(part_segm_fn)
        vertices_labels = dbw.compute_vertices_label(my_vertices_np, my_faces_np, part_segm_fn)

    no_check = [[1, 15, 16, 17, 18],
                [0, 2, 5, 8],
                [1, 3],
                [2, 4],
                [3],
                [1, 6],
                [5, 7],
                [6],
                [9, 12],
                [8, 10],
                [9, 11],
                [10, 22, 23, 24],
                [8, 13],
                [12, 14],
                [13, 19, 20, 21],
                [0, 16, 17, 18],
                [0, 15, 17, 18],
                [0, 15, 16, 18],
                [0, 15, 16, 17],
                [14, 20, 21],
                [14, 19, 21],
                [14, 19, 20],
                [11, 23, 24],
                [11, 22, 24],
                [11, 22, 23]]
    scan_proj = camera(scan_tensor).squeeze(0)
    gt_joints_3d = [None] * 25
    if kwargs.get('use_joint_loss_3d', False):
        for i in range(25):
            if joints_conf[0][i] < 0.5:
                continue
            flag = True
            for j in range(25):
                if joints_conf[0][j] < 0.5 or j in no_check[i] or j == i:
                    continue
                dis = torch.norm(gt_joints[0][j] - gt_joints[0][i]) / \
                      torch.norm(gt_joints[0][1] - gt_joints[0][8])
                if dis < 0.2:
                    flag = False
                    break
            if flag:
                with torch.no_grad():
                    diff_norm = torch.norm(scan_proj - gt_joints[0][i], dim=1)
                    if torch.min(diff_norm, -1)[0] > 5.0:
                        continue
                    min_i = torch.min(diff_norm, -1)[1].item()
                gt_joints_3d[i] = scan_tensor[0][min_i]

    # ****************************************
    # 4.load sdf of 3d scene (sdf directory)
    # (1) used in body-scene inter-penetration loss
    # ****************************************
    sdf = None
    sdf_normals = None
    grid_min = None
    grid_max = None
    voxel_size = None
    if sdf_penetration or joint_penetration:
        with open(osp.join(sdf_dir, scene_name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
            grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
            grid_dim = sdf_data['dim']
        voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(osp.join(sdf_dir, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = torch.tensor(sdf, dtype=dtype, device=device)
        if osp.exists(osp.join(sdf_dir, scene_name + '_normals.npy')):
            sdf_normals = np.load(osp.join(sdf_dir, scene_name + '_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
            sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)
        else:
            print("Normals not found")

    # ****************************************
    # 5.load the translation matrix from camera to world (cam2world directory)
    # ****************************************
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        cam2world = np.array(json.load(f))
        R = torch.tensor(cam2world[:3, :3].reshape(3, 3), dtype=dtype, device=device)
        t = torch.tensor(cam2world[:3, 3].reshape(1, 3), dtype=dtype, device=device)

    # ****************************************
    # 6.create the search tree and surface filter
    # (1) used in self-penetration loss
    # ****************************************
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), 'No CUDA Device! Interpenetration term can only be used' + ' with CUDA'

        # 1.BVH to search for the self intersection
        search_tree = BVH(max_collisions=max_collisions)

        # 2.forward function that returns the penetration loss
        pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=df_cone_height,
                                                                    point2plane=point2plane,
                                                                    vectorized=True,
                                                                    penalize_outside=penalize_outside)
        # 3.surface fileter to filter collision pairs
        if part_segm_fn:
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file, encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            filter_faces = FilterFaces(faces_segm=faces_segm,
                                       faces_parents=faces_parents,
                                       ign_part_pairs=ign_part_pairs).to(device=device)

    # ****************************************
    # 7.load vertex ids of contact parts (body_segments directory)
    # (1) used in contact loss
    # ****************************************
    contact_verts_ids = ftov = None
    if contact:
        # 1.human body contact vertice index
        contact_verts_ids = []
        for part in contact_body_parts:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
        contact_verts_ids = np.concatenate(contact_verts_ids)

        # 2.human body vertices (Nx3)
        vertices = body_model(return_verts=True,
                              body_pose=torch.zeros((batch_size, 63), dtype=dtype, device=device)).vertices
        vertices_np = vertices.detach().cpu().numpy().squeeze()

        # 3.human body triangular faces
        body_faces_np = body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3)

        # 4.humen body mesh
        m = Mesh(v=vertices_np, f=body_faces_np)

        ftov = m.faces_by_vertex(as_sparse_matrix=True)
        ftov = sparse.coo_matrix(ftov)

        indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
        values = torch.FloatTensor(ftov.data).to(device)
        shape = ftov.shape

        ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

    # ****************************************
    # 8.read the scene scan (scenes directory)
    # (1) used in body-scene inter-penetration loss
    # ****************************************
    scene_v = scene_vn = scene_f = None
    if scene_name is not None:
        if load_scene:
            scene = Mesh(filename=os.path.join(scene_dir, scene_name + '.ply'))
            scene.vn = scene.estimate_vertex_normals()
            # scene vertices
            scene_v = torch.tensor(scene.v[np.newaxis, :],
                                   dtype=dtype,
                                   device=device).contiguous()
            # scene vertice normals
            scene_vn = torch.tensor(scene.vn[np.newaxis, :],
                                    dtype=dtype,
                                    device=device)
            # scene faces
            scene_f = torch.tensor(scene.f.astype(int)[np.newaxis, :],
                                   dtype=torch.long,
                                   device=device)

    # ****************************************
    # 9.gather the weights
    # ****************************************
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    opt_weights_dict['joints_3d_weight'] = joints_3d_weight
    opt_weights_dict['joint_penetration_weight'] = joint_penetration_weight
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights
    if sdf_penetration:
        opt_weights_dict['sdf_penetration_weight'] = sdf_penetration_weights
    if contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    # weights -> tensor weights
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # ****************************************
    # 10.load body model's head indices (body_segments directory)
    # (1) used in m2s_dist
    # ****************************************
    with open(osp.join(body_segments_dir, 'body_mask.json'), 'r') as fp:
        head_indx = np.array(json.load(fp))
    N = body_model.get_num_verts()
    body_indx = np.setdiff1d(np.arange(N), head_indx)
    head_mask = np.in1d(np.arange(N), head_indx)
    body_mask = np.in1d(np.arange(N), body_indx)

    # joint indices used to compute the camera_loss
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    # ****************************************
    # 11.compute the body's initial translation matrix
    # (1) initialization mode: mean of the scan, similar triangles or the average of both
    # (2) if use depth image
    #       mode = scan
    #     else
    #       mode = similar triangles
    # ****************************************
    edge_indices = kwargs.get('body_tri_idxs')
    if init_mode == 'scan':
        init_t = init_trans
    elif init_mode == 'both':
        init_t = (init_trans.to(device) + fitting.guess_init(body_model,
                                                             gt_joints,
                                                             edge_indices,
                                                             use_vposer=use_vposer,
                                                             vposer=vposer,
                                                             pose_embedding=pose_embedding,
                                                             model_type=kwargs.get('model_type', 'smpl'),
                                                             focal_length=focal_length_x,
                                                             dtype=dtype)) / 2.0
    else:
        init_t = fitting.guess_init(body_model,
                                    gt_joints,
                                    edge_indices,
                                    use_vposer=use_vposer,
                                    vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length_x,
                                    dtype=dtype)

    # ****************************************
    # 12.create loss
    # ****************************************
    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      camera_mode=camera_mode,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face,
                               use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               s2m=s2m,
                               m2s=m2s,
                               rho_s2m=rho_s2m,
                               rho_m2s=rho_m2s,
                               head_mask=head_mask,
                               body_mask=body_mask,
                               sdf_penetration=sdf_penetration,
                               voxel_size=voxel_size,
                               grid_min=grid_min,
                               grid_max=grid_max,
                               sdf=sdf,
                               sdf_normals=sdf_normals,
                               R=R,
                               t=t,
                               contact=contact,
                               contact_verts_ids=contact_verts_ids,
                               rho_contact=rho_contact,
                               contact_angle=contact_angle,
                               dtype=dtype,
                               scan_labels=scan_labels,
                               vertices_labels=vertices_labels,
                               joint_penetration=joint_penetration,
                               **kwargs)
    loss = loss.to(device=device)

    # ****************************************
    # 13.fit
    # ****************************************
    with fitting.FittingMonitor(batch_size=batch_size,
                                visualize=visualize,
                                viz_mode=viz_mode,
                                **kwargs) as monitor:
        img = torch.tensor(img, dtype=dtype)
        H, W, _ = img.shape

        # ****************************************
        # 1.set the camera parameters to be optimized
        # (1) PROX use 'fixed' mode by default
        # (2) global_orient: direction of human body
        # ****************************************
        if camera_mode == 'moving':
            body_model.reset_params(body_pose=body_mean_pose)
            with torch.no_grad():
                camera.translation[:] = init_t.view_as(camera.translation)
                camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5
            camera.translation.requires_grad = True
            camera_opt_params = [camera.translation,
                                 body_model.global_orient]
        elif camera_mode == 'fixed':
            body_model.reset_params(body_pose=body_mean_pose,
                                    transl=init_t)
            camera_opt_params = [body_model.transl,
                                 body_model.global_orient]

        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # ****************************************
        # 2.optimize 'camera_opt_params'
        # ****************************************
        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(camera_opt_params, **kwargs)
        fit_camera = monitor.create_fitting_closure(camera_optimizer,
                                                    body_model,
                                                    camera,
                                                    gt_joints,
                                                    camera_loss,
                                                    create_graph=camera_create_graph,
                                                    use_vposer=use_vposer,
                                                    vposer=vposer,
                                                    pose_embedding=pose_embedding,
                                                    scan_tensor=scan_tensor,
                                                    return_full_pose=False,
                                                    return_verts=False)
        camera_init_start = time.time()

        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params,
                                                body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('\nCamera initialization done after {:.4f}'.format(time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}\n'.format(cam_init_loss_val))

        # ****************************************
        # 3.orientations to fit in
        # (1) If the 2D detections/positions of the shoulder joints are too close,
        # then rotate the body by 180 degrees and also fit to that orientation
        # (2) may be the man is facing left or right
        # ****************************************
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].\
                dot(cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations and
        # pick the orientation resulting in the lowest error
        results = []

        body_transl = body_model.transl.clone().detach()

        if use_ini_pose:
            if use_vposer:
                if use_internal:
                    vposer_encoder = VPoserEncoder(vposer_ckpt=vposer_ckpt,
                                                   latent_dim=vposer_latent_dim,
                                                   dtype=dtype,
                                                   **kwargs)
                    vposer_encoder.eval()
                    ini_pose = tgm.angle_axis_to_rotation_matrix(
                        ini_pose.reshape(-1, 3))[:, :3, :3]. \
                        contiguous().view(batch_size, -1)
                    ini_pose_embedding = vposer_encoder(
                        ini_pose).detach().numpy()
                else:
                    ini_pose_embedding = vposer.encode(
                        ini_pose).detach().numpy()
                pose_embedding = torch.tensor(ini_pose_embedding,
                                              dtype=dtype,
                                              device=device,
                                              requires_grad=True)
            else:
                ini_pose = ini_pose.detach().cpu()

        # ****************************************
        # 4.optimize the body
        # ****************************************
        final_loss_val = 0
        # (1) optimize in defferent orientations
        # for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
        for or_idx, orient in enumerate(orientations):
            opt_start = time.time()

            # (2) reset params in the beginning
            if use_ini_pose:
                new_params = defaultdict(transl=body_transl,
                                         global_orient=orient,
                                         body_pose=ini_pose)
            else:
                new_params = defaultdict(transl=body_transl,
                                         global_orient=orient,
                                         body_pose=body_mean_pose)
                if use_vposer:
                    with torch.no_grad():
                        pose_embedding.fill_(0)
            body_model.reset_params(**new_params)

            # (3) optimize using different weights in different stages
            # for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
            for opt_idx, curr_weights in enumerate(opt_weights):
                # 1.params to be optimized
                if opt_idx not in trans_opt_stages:
                    body_model.transl.requires_grad = False
                else:
                    body_model.transl.requires_grad = True

                body_params = list(body_model.parameters())
                final_params = list(filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                # 2.optimizer
                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)
                body_optimizer.zero_grad()

                # 3.weights
                curr_weights['bending_prior_weight'] = 3.17 * curr_weights['body_pose_weight']
                if use_hands:
                    joint_weights[:, 25:76] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 76:] = curr_weights['face_weight']

                loss.reset_loss_weights(curr_weights)

                # 4.closure
                closure = monitor.create_fitting_closure(body_optimizer,
                                                         body_model,
                                                         camera=camera,
                                                         gt_joints=gt_joints,
                                                         joints_conf=joints_conf,
                                                         joint_weights=joint_weights,
                                                         loss=loss,
                                                         create_graph=body_create_graph,
                                                         use_vposer=use_vposer,
                                                         vposer=vposer,
                                                         pose_embedding=pose_embedding,
                                                         scan_tensor=scan_tensor,
                                                         scene_v=scene_v,
                                                         scene_vn=scene_vn,
                                                         scene_f=scene_f,
                                                         ftov=ftov,
                                                         return_verts=True,
                                                         return_full_pose=True,
                                                         gt_joints_3d=gt_joints_3d)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()

                # 5.optimizing
                final_loss_val = monitor.run_fitting(body_optimizer,
                                                     closure,
                                                     final_params,
                                                     body_model,
                                                     pose_embedding=pose_embedding,
                                                     vposer=vposer,
                                                     use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    tqdm.write('Stage {} done after {:.4f} seconds\n'.format(opt_idx + 1, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write('Body fitting Orientation {} done after {:.4f} seconds'.format(or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}\n\n'.format(final_loss_val))

            # (4) get the fitting result
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})

            if use_vposer:
                result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
                if use_internal:
                    body_pose = vposer.forward(pose_embedding).view(1, -1) \
                        if use_vposer else None
                else:
                    body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1) \
                        if use_vposer else None
                result['body_pose'] = body_pose.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        # ****************************************
        # 5.save the parameters (.pkl)
        # ****************************************
        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    # ****************************************
    # 14.save the meshes (.ply)
    # ****************************************
    if save_meshes or visualize:
        if use_internal:
            body_pose = vposer.forward(pose_embedding).view(1, -1) \
                if use_vposer else None
        else:
            body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1) \
                if use_vposer else None

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = \
            model_type == 'smpl' and use_vposer
        if append_wrists:
            wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                     dtype=body_pose.dtype,
                                     device=body_pose.device)
            body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True,
                                  body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh
        out_mesh = trimesh.Trimesh(vertices,
                                   body_model.faces,
                                   process=False)
        out_mesh.export(mesh_fn)

    # ****************************************
    # 15.save the renderings (.png)
    # ****************************************
    if render_results:
        import pyrender

        # (1) initialization: camera, light and material
        H, W = 1080, 1920
        camera_center = np.array([951.30, 536.77])
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        camera = pyrender.camera.IntrinsicsCamera(fx=1060.53,
                                                  fy=1060.38,
                                                  cx=camera_center[0],
                                                  cy=camera_center[1])

        light = pyrender.DirectionalLight(color=np.ones(3),
                                          intensity=2.0)

        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
                                                      alphaMode='OPAQUE',
                                                      baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        # (2) render 3D human body on the RGB image
        # a.human body mesh
        body_mesh = pyrender.Mesh.from_trimesh(out_mesh,
                                               material=material)

        # b.pyrender scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_mesh, 'mesh')

        # c.renderer
        img = img.detach().cpu().numpy()
        H, W, _ = img.shape
        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)

        # d.render human body
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        # human body mask
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

        # e.save output image
        input_img = img
        # output_img = rendered human body + background image
        output_img = color[:, :, :-1] * valid_mask + \
                     input_img * (1 - valid_mask)
        img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        img.save(out_img_fn)

        # 3.render human body and static scene
        # a.human body mesh
        body_mesh = pyrender.Mesh.from_trimesh(out_mesh,
                                               material=material)

        # b.static scene mesh
        static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.ply'))
        # world -> camera translation matrix
        trans = np.linalg.inv(cam2world)
        # static_scene: world -> camera coordinate system
        static_scene.apply_transform(trans)
        # static scene
        static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

        # c.pyrender scene
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(static_scene_mesh, 'mesh')
        scene.add(body_mesh, 'mesh')

        # d.renderer
        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H)

        # e.render human body and static scene
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0

        # f.save output image
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        img.save(body_scene_rendering_fn)
