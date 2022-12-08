from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
import trimesh
from skimage import measure
import numpy as np
import pyrender


scene_path = 'BasementSittingBooth.ply'
mesh = trimesh.load(scene_path)


# voxels = mesh_to_voxels(mesh, 64, pad=True)
# vertices, faces, normals, _ = measure.marching_cubes_lewiner(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()


points, sdf = sample_sdf_near_surface(mesh, number_of_points=2500000)
colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)