# -*- coding: utf-8 -*-

import os
import pickle
import open3d as o3d
import torch
import smplx
import numpy as np

from cmd_parser import parse_config

new_folder = '../../Data/PROX/output-4'

gt_folder = '../../Data/PROX/quantitative/fittings/mosh'
ori_folder = '../../Data/PROX/output'
model_folder = '../../Data/PROX/models'
vposer_ckpt = '../../Data/PROX/vposerDecoderWeights.npz'
vposer_latent_dim = 63


def create_model(args):
    model_params = dict(model_path=model_folder,
                        create_global_orient=True,
                        create_body_pose=False,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        **args)
    model = smplx.create(gender='male', **model_params)
    return model

def vertices_joints(pkl, gt, **args):
    pkl = open(pkl, 'rb')
    params = pickle.load(pkl, encoding='bytes')

    model = create_model(args)
    if gt:
        model = model(jaw_pose=torch.tensor(params[b'jaw_pose']),
                      body_pose=torch.tensor(params[b'body_pose']),
                      right_hand_pose=torch.tensor(params[b'right_hand_pose']),
                      left_hand_pose=torch.tensor(params[b'left_hand_pose']))
    else:
        model = model(jaw_pose=torch.tensor(params['jaw_pose']),
                      body_pose=torch.tensor(params['body_pose']),
                      right_hand_pose=torch.tensor(params['right_hand_pose']),
                      left_hand_pose=torch.tensor(params['left_hand_pose']))

    vertices = model.vertices.detach().cpu().numpy().squeeze()
    joints = model.joints.detach().cpu().numpy().squeeze()

    # vis = o3d.Visualizer()
    # vis.create_window()
    # o3d_points = o3d.PointCloud()
    # o3d_points.points = o3d.Vector3dVector(vertices)
    # vis.add_geometry(o3d_points)
    # vis.run()

    return vertices, joints

def distance(gt, ori_new):
    points_num = gt.shape[0]
    sum = 0.0
    for i in range(points_num):
        sum += np.linalg.norm(gt[i] - ori_new[i])
    sum = sum / points_num
    return sum

def evaluate_frame(scene, frame_id, **args):
    gt_results_folder = gt_folder + '/' + scene + '/results'
    frame = sorted(os.listdir(gt_results_folder))[frame_id-1]

    gt_pkl = gt_results_folder + '/' + frame + '/000.pkl'
    gt_vertices, gt_joints = vertices_joints(gt_pkl, True, **args)

    ori_results_folder = ori_folder +  '/' + scene + '/results'
    ori_pkl = ori_results_folder + '/' + frame + '/000.pkl'
    ori_vertices, ori_joints = vertices_joints(ori_pkl, False, **args)
    ori_mpve = distance(gt_vertices, ori_vertices)
    ori_mpje = distance(gt_joints, ori_joints)

    new_results_folder = new_folder +  '/' + scene + '/results'
    new_pkl = new_results_folder + '/' + frame + '/000.pkl'
    new_vertices, new_joints = vertices_joints(new_pkl, False, **args)
    new_mpve = distance(gt_vertices, new_vertices)
    new_mpje = distance(gt_joints, new_joints)

    return ori_mpve, ori_mpje, new_mpve, new_mpje

def evaluate_scene(scene, **args):
    gt_results_folder = gt_folder + '/' + scene + '/results'
    frames = sorted(os.listdir(gt_results_folder))
    ori_mpve = []; ori_mpje = []
    new_mpve = []; new_mpje = []
    for frame_id in range(len(frames)):
        frame_ori_mpve, frame_ori_mpje, frame_new_mpve, frame_new_mpje = evaluate_frame(scene, frame_id+1, **args)
        ori_mpve.append(frame_ori_mpve)
        ori_mpje.append(frame_ori_mpje)
        new_mpve.append(frame_new_mpve)
        new_mpje.append(frame_new_mpje)

    return ori_mpve, ori_mpje, new_mpve, new_mpje


if __name__ == "__main__":
    args = parse_config()
    args.pop('gender')

    scene_frame = True
    if scene_frame:
        ori_mpve, ori_mpje, new_mpve, new_mpje = evaluate_scene('vicon_03301_04', **args)
        mean_ori_mpve = np.mean(ori_mpve)
        mean_ori_mpje = np.mean(ori_mpje)
        mean_new_mpve = np.mean(new_mpve)
        mean_new_mpje = np.mean(new_mpje)
    else:
        ori_mpve, ori_mpje, new_mpve, new_mpje = evaluate_frame('vicon_03301_01', 10, **args)
