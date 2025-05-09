import open3d as o3d
import os
import numpy as np
import data.objects
import results.runs_sdf



# 获取当前目录下所有的 .obj 文件
obj_dir = os.path.join(os.path.dirname(results.runs_sdf.__file__),"03_04_155918/meshes_training/")
#obj_dir = os.path.dirname(data.objects.__file__)

obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]


# 读取所有的 obj 文件
meshes = []
for obj_file in obj_files:
    full_path = os.path.join(obj_dir, obj_file)
    mesh = o3d.io.read_triangle_mesh(full_path)
    if mesh.is_empty():
        print(f"警告: {full_path} 为空或无效，已跳过")
        continue

    # 生成随机颜色 (RGB，每个值在0-1之间)
    random_color = np.random.rand(3)
    
    # 将随机颜色应用到整个网格
    mesh.paint_uniform_color(random_color)
    
    meshes.append(mesh)

# 渲染所有模型，保持原始坐标
if meshes:
    o3d.visualization.draw_geometries(meshes)
else:
    print("没有可用的 .obj 文件进行渲染")
