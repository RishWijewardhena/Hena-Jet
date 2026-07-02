import open3d as o3d

bunny = o3d.data.BunnyMesh() #
mesh= o3d.io.read_triangle_mesh(bunny.path)
mesh.compute_vertex_normals()

pcd= mesh.sample_points_poisson_disk(number_of_points=2000)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

alpha = 0.03
print("Alpha shape with alpha =", alpha)
mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh_alpha.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_alpha], mesh_show_back_face=True)