import math
import numpy as np
import pickle
import random
import open3d as o3d
import sklearn.cluster as skc
import torch
import pickle as pkl

smplx2pascal = [5, 5, 5,
                2,
                6, 6,
                2,
                6, 6,
                2,
                6, 6,
                1,
                2, 2,
                1,
                3, 3, 4, 4, 4, 4]

def distance_2d(position1, position2):
    return ((position1[0] - position2[0]) ** 2 +
            (position1[1] - position2[1]) ** 2) ** 0.5

def compute_points_label(points_2d, keypoints, hp_fn):
    point_num = points_2d.shape[0]
    label_num = keypoints.shape[0]
    with open(hp_fn, 'rb') as f:
        hp =pkl.load(f, encoding='bytes')

    '''
    # arm
    keypoints[2] = (keypoints[2] + keypoints[3]) / 2
    keypoints[3] = (keypoints[3] + keypoints[4]) / 2
    keypoints[5] = (keypoints[5] + keypoints[6]) / 2
    keypoints[6] = (keypoints[6] + keypoints[7]) / 2
    # leg
    keypoints[9] = (keypoints[9] + keypoints[10]) / 2
    keypoints[10] = (keypoints[10] + keypoints[11]) / 2
    keypoints[12] = (keypoints[12] + keypoints[13]) / 2
    keypoints[13] = (keypoints[13] + keypoints[14]) / 2
    # center
    center = (keypoints[1] + keypoints[8]) / 2

    labels = [-3] * point_num
    for i in range(point_num):
        min_dis = float('inf')
        for l in range(label_num):
            dis = distance_2d(points_2d[i], keypoints[l])
            if dis < min_dis:
                min_dis = dis
                labels[i] = l

        center_dis = distance_2d(points_2d[i], center)
        if center_dis < min_dis:
            labels[i] = -3

    # label: 0-24
    '''

    labels = [-3] * point_num
    for i in range(point_num):
        x = int(points_2d[i][0])
        y = int(points_2d[i][1])
        labels[i] = hp[y][x]

    # 0-6
    return np.array(labels)

def compute_vertices_label(vertices, faces, smplx_parts_segm):
    with open(smplx_parts_segm, 'rb') as f:
        face_segm_data = pickle.load(f, encoding='latin1')
    faces_segm = face_segm_data['segm']

    vertices_labels = np.array([-1] * vertices.shape[0])
    for i in range(faces.shape[0]):
        for vi in range(3):
            if vertices_labels[faces[i][vi]] == -1:
                vertices_labels[faces[i][vi]] = faces_segm[i]

    # for i in range(vertices_labels.shape[0]):
    #     print('part: ', i)
    #     labels_color = gen_random_colors(vertices_labels.shape[0], 1, i)
    #     visualize_points_with_labels(vertices,
    #                                  vertices_labels,
    #                                  labels_color)

    # smplx (0-54) -> pascal0-6
    for i in range(len(vertices_labels)):
        if vertices_labels[i] >= 0 and vertices_labels[i] <= 21:
            vertices_labels[i] = smplx2pascal[vertices_labels[i]]
        elif vertices_labels[i] >= 22 and vertices_labels[i] <= 24:
            vertices_labels[i] = 1
        elif vertices_labels[i] >= 25 and vertices_labels[i] <= 54:
            vertices_labels[i] = 4

    return vertices_labels

def gen_random_colors(color_num, visualize_mode=2, label=-1):
    if visualize_mode == 1 and label != -1:
        random_colors = [[1.0, 0.0, 0.0]] * color_num
        random_colors[label] = [0.0, 0.0, 0.0]
    else:
        random.seed(0)
        random_colors = [[0.0, 0.0, 0.0]]
        while len(random_colors) < color_num:
            color = [random.random(), random.random(), random.random()]
            if color in random_colors:
                continue
            else:
                random_colors.append(color)

    return random_colors

def visualize_points_with_labels(points, labels, labels_color):
    colors = []
    for i in range(points.shape[0]):
        colors.append(labels_color[labels[i]])

    vis = o3d.Visualizer()
    vis.create_window()
    o3d_points = o3d.PointCloud()
    o3d_points.points = o3d.Vector3dVector(points)
    o3d_points.colors = o3d.Vector3dVector(colors)
    vis.add_geometry(o3d_points)
    vis.run()

def drop_noise(points, eps, min_samples):
    db = skc.DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(points)
    db_labels = db.labels_

    big_i = 0
    big_c = len(points[db_labels[:] == big_i])
    for i in range(len(np.unique(db_labels))):
        c = len(points[db_labels[:] == i])
        if c > big_c:
            big_c = c
            big_i = i

    print("\npoint number (before): {}".format(len(points)))
    print("noise number: {}".format(len(points[db_labels[:] == -1])))
    print("point number (biggest cluster): {}".format(len(points[db_labels[:] == big_i])))
    print("point number (after): {}\n".format(len(points[db_labels[:] != -1])))

    visualze = True
    if visualze:
        vis = o3d.Visualizer()
        vis.create_window()
        o3d_points = o3d.PointCloud()
        o3d_points.points = o3d.Vector3dVector(points)
        vis.add_geometry(o3d_points)
        vis.run()

        vis = o3d.Visualizer()
        vis.create_window()
        o3d_points = o3d.PointCloud()
        o3d_points.points = o3d.Vector3dVector(points[db_labels[:] == big_i])
        vis.add_geometry(o3d_points)
        vis.run()

        vis = o3d.Visualizer()
        vis.create_window()
        o3d_points = o3d.PointCloud()
        o3d_points.points = o3d.Vector3dVector(points[db_labels[:] != -1])
        vis.add_geometry(o3d_points)
        vis.run()

    return db_labels[:] == big_i

def get_degree(a, b, c):
    ba = a - b
    bc = c - b
    deg1 = math.atan2(ba[1], ba[0])
    deg2 = math.atan2(bc[1], bc[0])
    if deg1 * deg2 >= 0:
        deg = abs(deg1 - deg2)
    else:
        deg = abs(deg1) + abs(deg2)
        if deg > math.pi:
            deg = 2 * math.pi - deg

    return deg

def get_ini_pose(keypoints, batch_size, body_joints_num, dtype):
    body_keypoints = keypoints[0, :25]

    l_leg = 0
    l_a = body_keypoints[12]
    l_b = body_keypoints[13]
    l_c = body_keypoints[14]
    if l_a[2] > 0 and l_b[2] > 0 and l_c[2] > 0:
        l_leg = math.pi - get_degree(l_a[:2], l_b[:2], l_c[:2])

    r_leg = 0
    r_a = body_keypoints[9]
    r_b = body_keypoints[10]
    r_c = body_keypoints[11]
    if r_a[2] > 0 and r_b[2] > 0 and r_c[2] > 0:
        r_leg = math.pi - get_degree(r_a[:2], r_b[:2], r_c[:2])

    ini_pose = torch.zeros([batch_size, body_joints_num * 3], dtype=dtype)
    ini_pose[0, 0 * 3 + 0] = -l_leg
    ini_pose[0, 3 * 3 + 0] = l_leg
    ini_pose[0, 1 * 3 + 0] = -r_leg
    ini_pose[0, 4 * 3 + 0] = r_leg

    return ini_pose
