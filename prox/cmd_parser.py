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

import os
import configargparse


def parse_config():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of PROX'

    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='PROX')

    # ****************************************
    # unused
    # ****************************************
    parser.add_argument('--max_persons',
                        type=int,
                        default=3,
                        help='The maximum number of persons to process')
    parser.add_argument('--viz_mode',
                        type=str,
                        default='o3d',
                        choices=['mv', 'o3d'],
                        help='')
    parser.add_argument('--degrees',
                        type=float,
                        default=[0, 90, 180, 270],
                        help='Degrees of rotation for rendering the final result')
    parser.add_argument('--joints_to_ign',
                        type=int,
                        default=-1,
                        nargs='*',
                        help='Indices of joints to be ignored')
    parser.add_argument('--optim_jaw',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='Optimize over the jaw pose')
    parser.add_argument('--optim_hands',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='Optimize over the hand pose')
    parser.add_argument('--optim_expression',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='Optimize over the expression')
    parser.add_argument('--optim_shape',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='Optimize over the shape space')
    parser.add_argument('--flat_hand_mean',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Use the flat hand as the mean pose')
    parser.add_argument('--init_joints_idxs',
                        type=int,
                        nargs='*',
                        default=[9, 12, 2, 5],
                        help='Which joints to use for initializing the camera')
    parser.add_argument('--body_tri_idxs',
                        type=lambda x: [list(map(int, pair.split('.')))
                                        for pair in x.split(',')],
                        default='5.12,2.9',
                        help='The indices of the joints used to estimate' +
                        ' the initial depth of the camera. The format' +
                        ' should be vIdx1.vIdx2,vIdx3.vIdx4')

    parser.add_argument('--penalize_outside',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=False,
                        help='Penalize outside')
    parser.add_argument('--df_cone_height',
                        type=float,
                        default=0.5,
                        help='The default value for the height of the cone' +
                        ' that is used to calculate the penetration distance' +
                        ' field')
    parser.add_argument('--max_collisions',
                        type=int,
                        default=8,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--point2plane',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Use point to plane distance')
    parser.add_argument('--ign_part_pairs',
                        type=str,
                        default=None,
                        nargs='*',
                        help='Pairs of parts whose collisions will be ignored')
    parser.add_argument('--side_view_thsh',
                        type=float,
                        default=25,
                        help='This is thresholding value that determines' +
                        ' whether the human is captured in a side view.' +
                        'If the pixel distance between the shoulders is less' +
                        ' than this value, two initializations of SMPL fits' +
                        ' are tried.')
    parser.add_argument('--load_scene',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')


    # ****************************************
    # used
    # ****************************************
    parser.add_argument('--rho',
                        type=float,
                        default=100,
                        help='Value of constant of robust loss')
    parser.add_argument('--interpenetration',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='Whether to use the interpenetration term')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-6,
                        help='The learning rate for the algorithm')
    parser.add_argument('--gtol',
                        type=float,
                        default=1e-8,
                        help='The tolerance threshold for the gradient')
    parser.add_argument('--ftol',
                        type=float,
                        default=2e-9,
                        help='The tolerance threshold for the function')
    parser.add_argument('--maxiters',
                        type=int,
                        default=100,
                        help='The maximum iterations for the optimization')
    parser.add_argument('--frame_ids',
                        type=int,
                        default=None,
                        nargs='*',
                        help='')
    parser.add_argument('--start',
                        type=int,
                        default=1,
                        help='id of the starting frame')
    parser.add_argument('--step',
                        type=int,
                        default=1,
                        help='step')
    parser.add_argument('--flip',
                        default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='flip image and keypoints')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='The size of the batch')
    parser.add_argument('--num_gaussians',
                        default=8,
                        type=int,
                        help='The number of gaussian for the Pose Mixture Prior')
    parser.add_argument('--use_pca',
                        default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the low dimensional PCA space for the hands')
    parser.add_argument('--num_pca_comps',
                        default=6,
                        type=int,
                        help='The number of PCA components for the hand')
    parser.add_argument('--use_joints_conf',
                        default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the confidence scores for the optimization')
    parser.add_argument('--interactive',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')
    parser.add_argument('--save_meshes',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Save final output meshes')
    parser.add_argument('--visualize',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Display plots while running the optimization')
    parser.add_argument('--gender_lbl_type',
                        type=str,
                        default='none',
                        choices=['none', 'gt', 'pd'],
                        help='The type of gender label to use')
    parser.add_argument('--gender',
                        type=str,
                        default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL model')
    parser.add_argument('--model_type',
                        type=str,
                        default='smpl',
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of the model that we will fit to the data')
    parser.add_argument('--render_results',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='render final results')
    parser.add_argument('--trans_opt_stages',
                        type=int,
                        default=[2,3,4],
                        nargs='*',
                        help='stages where translation will be optimized')
    parser.add_argument('--use_vposer',
                        default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the VAE pose embedding')


    # ****************************************
    # fixed
    # ****************************************
    parser.add_argument('--use_cuda',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Use CUDA for the computations')
    parser.add_argument('--dataset',
                        type=str,
                        default='hands_cmu_gt',
                        help='The name of the dataset that will be used')
    parser.add_argument('--float_dtype',
                        type=str,
                        default='float32',
                        help='The types of floats used')
    parser.add_argument('--loss_type',
                        default='smplify',
                        type=str,
                        help='The type of loss to use')
    parser.add_argument('--optim_type',
                        type=str,
                        default='adam',
                        help='The optimizer used')


    # ****************************************
    # hand & face
    # ****************************************
    parser.add_argument('--use_hands',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the hand keypoints in the SMPL optimization process')
    parser.add_argument('--use_face',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the facial keypoints in the optimization process')
    parser.add_argument('--use_face_contour',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the dynamic contours of the face')


    # ****************************************
    # prior type
    # ****************************************
    parser.add_argument('--body_prior_type',
                        type=str,
                        default='mog',
                        help='The type of prior that will be used to' +
                        ' regularize the optimization. Can be a Mixture of Gaussians (mog)')
    parser.add_argument('--left_hand_prior_type',
                        type=str,
                        default='mog',
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' left hand. Can be a Mixture of Gaussians (mog)')
    parser.add_argument('--right_hand_prior_type',
                        type=str,
                        default='mog',
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' right hand. Can be a Mixture of Gaussians (mog)')
    parser.add_argument('--jaw_prior_type',
                        type=str,
                        default='l2',
                        choices=['l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the jaw')


    # ****************************************
    # weights
    # ****************************************
    parser.add_argument('--data_weights',
                        nargs='*',
                        default=[1, ] * 5,
                        type=float,
                        help='The weight of the data term')
    parser.add_argument('--body_pose_prior_weights',
                        default=[4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                        nargs='*',
                        type=float,
                        help='The weights of the body pose regularizer')
    parser.add_argument('--shape_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float,
                        nargs='*',
                        help='The weights of the Shape regularizer')
    parser.add_argument('--expr_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float,
                        nargs='*',
                        help='The weights of the Expressions regularizer')
    parser.add_argument('--face_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0],
                        type=float,
                        nargs='*',
                        help='The weights for the facial keypoints' +
                        ' for each stage of the optimization')
    parser.add_argument('--hand_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0],
                        type=float,
                        nargs='*',
                        help='The weights for the 2D joint error of the hands')
    parser.add_argument('--jaw_pose_prior_weights',
                        nargs='*',
                        help='The weights of the pose regularizer of the hands')
    parser.add_argument('--hand_pose_prior_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float,
                        nargs='*',
                        help='The weights of the pose regularizer of the hands')
    parser.add_argument('--coll_loss_weights',
                        default=[0.0, 0.0, 0.0, 2.0],
                        type=float,
                        nargs='*',
                        help='The weight for the collision term')
    parser.add_argument('--depth_loss_weight',
                        default=1e2,
                        type=float,
                        help='The weight for the regularizer for the' +
                        ' z coordinate of the camera translation')


    # ****************************************
    # path
    # ****************************************
    parser.add_argument('--config',
                        required=True,
                        is_config_file=True,
                        help='config file path')
    parser.add_argument('--recording_dir',
                        default=os.getcwd(),
                        help='The directory that contains the data')
    parser.add_argument('--output_folder',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--img_folder',
                        type=str,
                        default='Color',
                        help='The folder where the images are stored')
    parser.add_argument('--summary_folder',
                        type=str,
                        default='summaries',
                        help='Where to store the TensorBoard summaries')
    parser.add_argument('--result_folder',
                        type=str,
                        default='results',
                        help='The folder with the pkls of the output parameters')
    parser.add_argument('--mesh_folder',
                        type=str,
                        default='meshes',
                        help='The folder where the output meshes are stored')
    parser.add_argument('--prior_folder',
                        type=str,
                        default='prior',
                        help='The folder where the prior is stored')
    parser.add_argument('--model_folder',
                        default='models',
                        type=str,
                        help='The directory where the models are stored')
    parser.add_argument('--mask_folder',
                        type=str,
                        default='BodyIndex',
                        help='The folder where the keypoints are stored')
    parser.add_argument('--part_segm_fn',
                        default='',
                        type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--vposer_ckpt',
                        type=str,
                        default='',
                        help='The path to the V-Poser checkpoint')


    # ****************************************
    # camera
    # ****************************************
    parser.add_argument('--camera_type',
                        type=str,
                        default='persp',
                        choices=['persp'],
                        help='The type of camera used')
    parser.add_argument('--camera_mode',
                        type=str,
                        default='moving',
                        choices=['moving', 'fixed'],
                        help='The mode of camera used')
    parser.add_argument('--focal_length_x',
                        type=float,
                        default=5000,
                        help='Value of focal length.')
    parser.add_argument('--focal_length_y',
                        type=float,
                        default=5000,
                        help='Value of focal length.')
    parser.add_argument('--camera_center_x',
                        type=float,
                        default=None,
                        help='Value of camera center x.')
    parser.add_argument('--camera_center_y',
                        type=float,
                        default=None,
                        help='Value of camera center y.')


    # ****************************************
    # depth loss
    # ****************************************
    parser.add_argument('--init_mode',
                        type=str,
                        default=None,
                        choices=[None, 'scan', 'both'],
                        help='')
    parser.add_argument('--s2m_weights',
                        type=float,
                        default=[0.0, 0.0, 0.0, 0.0, 0.0],
                        nargs='*',
                        help='')
    parser.add_argument('--s2m',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Whether to save the meshes')
    parser.add_argument('--m2s',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Whether to save the meshes')
    parser.add_argument('--m2s_weights',
                        type=float,
                        default=[0.0, 0.0, 0.0, 0.0, 0.0],
                        nargs='*',
                        help='')
    parser.add_argument('--rho_s2m',
                        type=float,
                        default=1,
                        help='Value of constant of robust loss')
    parser.add_argument('--rho_m2s',
                        type=float,
                        default=1,
                        help='Value of constant of robust loss')
    parser.add_argument('--read_depth',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Read depth frames')
    parser.add_argument('--read_mask',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='Read masks')
    parser.add_argument('--mask_on_color',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')


    # ****************************************
    # body-scene penetration loss
    # ****************************************
    parser.add_argument('--sdf_penetration',
                        default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='')
    parser.add_argument('--sdf_penetration_weights',
                        default=[0.0, 0.0, 0.0, 0.0, 0.0],
                        nargs='*',
                        type=float,
                        help='')


    # ****************************************
    # contact loss
    # ****************************************
    parser.add_argument('--contact',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--rho_contact',
                        type=float,
                        default=1,
                        help='Value of constant of robust loss')
    parser.add_argument('--contact_angle',
                        type=float,
                        default=45,
                        help='used to refine normals. (angle in degrees)')
    parser.add_argument('--contact_loss_weights',
                        type=float,
                        default=[0.0, 0.0, 0.0, 0.0, 0.0],
                        nargs='*',
                        help='The weight for the contact term')
    parser.add_argument('--contact_body_parts',
                        type=str,
                        default=['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs'],
                        nargs='*',
                        help='')


    # ****************************************
    # dangbowen
    # ****************************************
    parser.add_argument('--drop_noise',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--eps',
                        default=0.2,
                        help='')
    parser.add_argument('--min_samples',
                        default=100,
                        help='')
    parser.add_argument('--new_depth_loss',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--use_internal',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='')
    parser.add_argument('--use_ini_pose',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--use_joints_loss_3d',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--joints_3d_weight',
                        type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0],
                        nargs='*',
                        help='')
    parser.add_argument('--joint_penetration',
                        type=lambda x: x.lower() in ['true', '1'],
                        default=False,
                        help='')
    parser.add_argument('--joint_penetration_weight',
                        type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0],
                        nargs='*',
                        help='')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
