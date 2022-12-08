import os
import os.path as osp
import cv2
import numpy as np
import json
import open3d as o3d
import argparse
import torch
import pickle
import smplx


def main(args):
    # output/SceneName_SubjectID_SequenceID
    fitting_dir = args.fitting_dir
    # SceneName_SubjectID_SequenceID
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    # Output/SceneName_SubjectID_SequenceID/results
    fitting_dir = osp.join(fitting_dir, 'results')
    # SceneName
    scene_name = recording_name.split("_")[0]
    # PROX/qualitative
    base_dir = args.base_dir
    # PROX/qualitative/cam2world
    cam2world_dir = osp.join(base_dir, 'cam2world')
    # PROX/qualitative/scenes
    scene_dir = osp.join(base_dir, 'scenes')
    # PROX/qualitative/recordings/SceneName_SubjectID_SequenceID
    recording_dir = osp.join(base_dir, 'recordings', recording_name)
    # PROX/qualitative/recordings/SceneName_SubjectID_SequenceID/Color
    color_dir = os.path.join(recording_dir, 'Color')

    female_subjects_ids = [162, 3452, 159, 3403]
    subject_id = int(recording_name.split('_')[1])
    if subject_id in female_subjects_ids:
        gender = 'female'
    else:
        gender = 'male'

    # 1.cv2 window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    # 2.o3d visualizer
    vis = o3d.Visualizer()
    vis.create_window()

    # 3.add <static scene mesh> & <human body mesh> to o3d visualizer
    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    body = o3d.TriangleMesh()
    vis.add_geometry(scene)
    vis.add_geometry(body)

    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        trans = np.array(json.load(f))

    model = smplx.create(args.model_folder,
                         model_type='smplx',
                         gender=gender,
                         ext='npz',
                         num_pca_comps=args.num_pca_comps,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True)

    count = 0

    # 4.visualize frame by frame
    for img_name in sorted(os.listdir(color_dir))[args.start-1::args.step]:
        img_name = img_name[:-4]
        print('viz frame {}'.format(img_name))

        # (1).human body params
        with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])

        # (2).human body
        output = model(return_verts=True, **torch_param)

        # (3).change human body mesh
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body.vertices = o3d.Vector3dVector(vertices)
        body.triangles = o3d.Vector3iVector(model.faces)
        body.vertex_normals = o3d.Vector3dVector([])
        body.triangle_normals = o3d.Vector3dVector([])
        body.compute_vertex_normals()
        # human body mesh: camera -> world
        body.transform(trans)

        # (4).RGB image
        color_img = cv2.imread(os.path.join(color_dir, img_name + '.jpg'))
        color_img = cv2.flip(color_img, 1)

        # (5).update o3d visualizer
        vis.update_geometry()

        # (6).visualize
        while True:
            cv2.imshow('frame', color_img)

            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(30)
            if key == 27:
                break

        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('fitting_dir',
                        type=str,
                        default=os.getcwd(),
                        help='recording dir')
    parser.add_argument('--base_dir',
                        type=str,
                        default='/media/dangbowen/Data/PROX/qualitative',
                        help='recording dir')
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='id of the starting frame')
    parser.add_argument('--step',
                        type=int,
                        default=1,
                        help='id of the starting frame')
    parser.add_argument('--model_folder',
                        default='/media/dangbowen/Data/PROX/models',
                        type=str,
                        help='')
    parser.add_argument('--num_pca_comps',
                        type=int,
                        default=12,
                        help='')
    parser.add_argument('--gender',
                        type=str,
                        default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                             'model')

    args = parser.parse_args()
    main(args)
