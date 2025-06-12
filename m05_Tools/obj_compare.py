import os

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import m02_Data_Files.d05_SDF_Results.runs_sdf
import m02_Data_Files.d02_Object_Files

#===========================
run_index = "11_06_171540"
#===========================

run_folder = run_index + "/meshes_training/"
# 获取当前目录下所有的 .obj 文件
obj_dir_base = os.path.join(os.path.dirname(m02_Data_Files.d05_SDF_Results.runs_sdf.__file__),run_folder)
Obj_dir_target = os.path.dirname(m02_Data_Files.d02_Object_Files.__file__)

obj_files_base = [f for f in os.listdir(obj_dir_base) if f.endswith(".obj")]
obj_files_target = [f for f in os.listdir(Obj_dir_target) if f.endswith(".obj")]

def append_meshes(obj_files, obj_path):
    meshes = []
    for obj_file in obj_files:
        full_path = os.path.join(obj_path, obj_file)
        mesh = o3d.io.read_triangle_mesh(full_path)
        if mesh.is_empty():
            print(f"警告: {full_path} 为空或无效，已跳过")
            continue
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    return meshes


# 渲染所有模型，保持原始坐标
if meshes:
    o3d.visualization.draw_geometries(meshes)
else:
    print("没有可用的 .obj 文件进行渲染")
