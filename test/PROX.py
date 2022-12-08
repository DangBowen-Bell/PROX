#%% import library & other

import numpy as np
import json
import cv2
import sys
import pickle
import random
import open3d as o3d
import os

sys.path.append('prox/')
import data_parser
from projection_utils import Projection

qualitative_base_dir = '../Data/PROX/qualitative/'
quantitative_base_dir = '../Data/PROX/quantitative/'

qualitative_projection = Projection(qualitative_base_dir+'calibration')
quantitative_projection = Projection(quantitative_base_dir+'calibration')

default_scene_name = 'BasementSittingBooth'
default_subject_sequence = '00142_01'
default_frame_id = 's001_frame_00001__00.00.00.029'

# qualitative dataset does not have sdf_normals file
sdf_normals_path = '../Data/PROX/quantitative/sdf/vicon_normals.npy'
smplx_parts_segm_path = '../Data/PROX/smplx_parts_segm.pkl'

# (118,3)
def get_keypoints(base_dir=qualitative_base_dir,
                  scene_name=default_scene_name,
                  subject_sequence=default_subject_sequence,
                  frame_id=default_frame_id):
    # Keypoints(keypoints=(1,118,3), gender_gt=[], gender_pd=[])
    keypoints = data_parser.read_keypoints(base_dir + 'keypoints/{0}_{1}/{2}_keypoints.json'.
                                           format(scene_name, subject_sequence, frame_id),
                                           use_hands=True,
                                           use_face=True)
    keypoints = keypoints[0][0]
    return keypoints

# (1080,1920)
def get_color_img(is_flip,
                  base_dir=qualitative_base_dir,
                  scene_name=default_scene_name,
                  subject_sequence=default_subject_sequence,
                  frame_id=default_frame_id):
    img = cv2.imread(base_dir + 'recordings/{0}_{1}/Color/{2}.jpg'.
                     format(scene_name, subject_sequence, frame_id)). \
              astype(np.float32)[:, :, ::-1] / 255.0
    if is_flip:
        img = cv2.flip(img, 1)
    return img

# (424,512)
def get_depth_img(is_flip,
                  base_dir=qualitative_base_dir,
                  scene_name=default_scene_name,
                  subject_sequence=default_subject_sequence,
                  frame_id=default_frame_id):
    depth_img = cv2.imread(base_dir + 'recordings/{0}_{1}/Depth/{2}.png'.
                           format(scene_name, subject_sequence, frame_id),
                           flags=-1). \
                    astype(float) / 8.
    depth_img /= 1000.0
    if is_flip:
        depth_img = cv2.flip(depth_img, 1)
    return depth_img

# (1080,1920)
def get_color_mask(is_flip,
                   base_dir=qualitative_base_dir,
                   scene_name=default_scene_name,
                   subject_sequence=default_subject_sequence,
                   frame_id=default_frame_id):
    color_mask = cv2.imread(base_dir + 'recordings/{0}_{1}/BodyIndexColor/{2}.png'.
                            format(scene_name, subject_sequence, frame_id),
                            cv2.IMREAD_GRAYSCALE)
    if is_flip:
       color_mask = cv2.flip(color_mask, 1)
    return color_mask

# (1080,1920)
def get_mask(is_flip,
             base_dir=qualitative_base_dir,
             scene_name=default_scene_name,
             subject_sequence=default_subject_sequence,
             frame_id=default_frame_id):
    mask = cv2.imread(base_dir + 'recordings/{0}_{1}/BodyIndex/{2}.png'.
                      format(scene_name, subject_sequence, frame_id),
                      cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
    if is_flip:
        mask = cv2.flip(mask, 1)
    return mask

def get_points_2d(depth_img, mask_on_color, mask, projection):
    # (217088,3)
    all_points = projection.unproject_depth_image(depth_img, projection.depth_cam)

    # (217088,2)
    uvs = projection.projectPoints(all_points, projection.color_cam)
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
        # 2D human body (25831,2) [column, row]
        uvs = uvs[valid_mask_idx == True]
    else:
        # 2D human body (24696,2) [column, row]
        uvs = uvs[valid_idx == True]
    return uvs

def gen_random_colors(color_num):
    random_colors = []
    while len(random_colors) < color_num:
        color = [random.random(), random.random(), random.random()]
        if color in random_colors:
            continue
        else:
            random_colors.append(color)
    return random_colors

def visualize_points(points, colors=None):
    vis = o3d.Visualizer()
    vis.create_window()
    o3d_points = o3d.PointCloud()
    o3d_points.points = o3d.Vector3dVector(points)
    if colors is not None:
        o3d_points.colors = o3d.Vector3dVector(colors)
    vis.add_geometry(o3d_points)
    vis.run()

#%% fit_single_frame.py: 4

with open(qualitative_base_dir+'sdf/{0}.json'.format(default_scene_name), 'r') as f:
    # {'max':(3,), 'dim':, 'min':(3,)}
    sdf_data = json.load(f)
    # (3,)
    grid_min = np.array(sdf_data['min'])
    # (3,)
    grid_max = np.array(sdf_data['max'])
    # 256
    grid_dim = sdf_data['dim']
voxel_size = (grid_max - grid_min) / grid_dim

# 256x256x256
sdf = np.load(qualitative_base_dir+'sdf/{0}_sdf.npy'.format(default_scene_name)).\
    reshape(grid_dim, grid_dim, grid_dim)
# 256x256x256
sdf_normals = np.load(sdf_normals_path).\
    reshape(grid_dim, grid_dim, grid_dim, 3)

points = []
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            if sdf[i, j, k] < 0:
                points.append([i, j, k])

vis = o3d.Visualizer()
vis.create_window()
o3d_points = o3d.PointCloud()
o3d_points.points = o3d.Vector3dVector(points)
vis.add_geometry(o3d_points)
vis.run()

#%% fit_single_frame.py: 6

with open(smplx_parts_segm_path, 'rb') as f:
    # {'segm':(20908,), 'parents':(20908,)}
    face_segm_data = pickle.load(f, encoding='latin1')
# (20908,)
faces_segm = face_segm_data['segm']
# (20908,)
faces_parents = face_segm_data['parents']

#%% fit_single_frame.py: 7

contact_verts_ids = []
for part in ['L_Hand', 'R_Leg']:
    with open(qualitative_base_dir+'body_segments/'+part+'.json', 'r') as f:
        # {'verts_ind':(97,), 'faces_ind':(135,)}
        data = json.load(f)
        contact_verts_ids.append(list(set(data['verts_ind'])))
# (358,)
contact_verts_ids = np.concatenate(contact_verts_ids)

#%% fit_single_frame.py: 9

weights_dict = {'aa':[-1,1], 'bb':[-2,2], 'cc':None, 'dd':[-4,4]}
# ['aa', 'bb', 'cc', 'dd']
keys = weights_dict.keys()
# [{'aa':-1, 'bb':-2, 'dd':-4}, {'aa':1, 'bb':2, 'dd':4}]
weights = [dict(zip(keys, vals)) for vals in
            zip(*(weights_dict[k] for k in keys
                  if weights_dict[k] is not None))]
# [{'aa':-1, 'bb':-2, 'dd':-4}, {'aa':1, 'bb':2, 'dd':4}]
for weight_list in weights:
    for key in weight_list:
        weight_list[key] = weight_list[key]

#%% fit_single_frame.py: 10

N = 10475
with open(qualitative_base_dir+'body_segments/body_mask.json', 'r') as fp:
    # (5023,)
    head_indx = np.array(json.load(fp))
# (10475,)
body_indx = np.setdiff1d(np.arange(N), head_indx)
# (10475,) -> [True, False, ...]
head_mask = np.in1d(np.arange(N), head_indx)
# (5452,) -> [True, False, ...]
body_mask = np.in1d(np.arange(N), body_indx)

#%% fit_single_frame.py: 15

# identity matrix
camera_pose = np.eye(4)
# [[1, 0, 0,0],
#  [0,-1, 0,0],
#  [0, 0,-1,0],
#  [0, 0, 0,1]]
camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose

#%% projection_utils.py

color_img = get_color_img(is_flip=True,
                          base_dir=quantitative_base_dir,
                          scene_name='vicon',
                          subject_sequence='03301_01',
                          frame_id='s001_frame_00001__00.00.00.023')
depth_img = get_depth_img(is_flip=True,
                          base_dir=quantitative_base_dir,
                          scene_name='vicon',
                          subject_sequence='03301_01',
                          frame_id='s001_frame_00001__00.00.00.023')

depth_raw = depth_img.copy()
depth_raw *= 8000.0

# color_img -> depth size
color_aligned = quantitative_projection.align_color2depth(depth_img,  color_img)
# cv2.namedWindow('color_aligned', cv2.WINDOW_NORMAL)
# cv2.imshow('color_aligned', color_aligned)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# depth_img -> color size
depth_aligned = quantitative_projection.align_depth2color(depth_img, depth_raw)
cv2.imwrite('test.png', depth_aligned)
cv2.namedWindow('depth_aligned', cv2.WINDOW_NORMAL)
cv2.imshow('depth_aligned', depth_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% dbw_utils.py

color_img = get_color_img(is_flip=True,
                          base_dir=quantitative_base_dir,
                          scene_name='vicon',
                          subject_sequence='03301_01',
                          frame_id='s001_frame_00001__00.00.00.023')
depth_img = get_depth_img(is_flip=True,
                          base_dir=quantitative_base_dir,
                          scene_name='vicon',
                          subject_sequence='03301_01',
                          frame_id='s001_frame_00001__00.00.00.023')
mask_on_color = True
mask = get_color_mask(is_flip=True,
                      base_dir=quantitative_base_dir,
                      scene_name='vicon',
                      subject_sequence='03301_01',
                      frame_id='s001_frame_00001__00.00.00.023') if mask_on_color else \
    get_mask(is_flip=True,
             base_dir=quantitative_base_dir,
             scene_name='vicon',
             subject_sequence='03301_01',
             frame_id='s001_frame_00001__00.00.00.023')

# mask_on_color = True: {'points':(25831,3), 'colors':(25831,3)}
# mask_on_color = False: {'points':(24696,3), 'colors':(24696,3)}
scan_dict = quantitative_projection.create_scan(mask, depth_img,
                                                mask_on_color=mask_on_color)
visualize_points(scan_dict['points'])

#%% smplx/body_models.py

model_path = '../Data/PROX/models/smplx'
gender = 'male'
ext = 'pkl'

model_fn = 'SMPLX_{}.{}'.format(gender.upper(), ext)
smplx_path = os.path.join(model_path, model_fn)

if ext == 'pkl':
    with open(smplx_path, 'rb') as smplx_file:
        model_data = pickle.load(smplx_file, encoding='latin1')
elif ext == 'npz':
    model_data = np.load(smplx_path, allow_pickle=True)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

data_struct = Struct(**model_data)
