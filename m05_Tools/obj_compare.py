import os
import sys
import numpy as np
import pyvista as pv
import tkinter as tk
from tkinter import filedialog, messagebox

def append_meshes(obj_files, obj_path, color,alpha=0.5):
    """
    加载 obj 文件并赋予随机颜色和透明度。
    """
    meshes = []
    for obj_file in obj_files:
        full_path = os.path.join(obj_path, obj_file)
        try:
            mesh = pv.read(full_path)
            
            mesh["colors"] = np.tile(color, (mesh.n_points, 1))
            meshes.append((mesh, color, alpha))
        except Exception as e:
            print(f"⚠️ 无法加载 {obj_file}: {e}")
    return meshes

def main(folder_1, folder_2):
    obj_files_1 = [f for f in os.listdir(folder_1) if f.endswith(".obj")]
    if folder_2 == None:
        color = np.random.rand(3)
        meshes = append_meshes(obj_files_1 , folder_1, color,alpha=0.7)
        if not meshes:
            print("❌ 没有可用的 .obj 文件进行渲染")
            return
        plotter = pv.Plotter()
        for mesh, color, alpha in meshes:
            plotter.add_mesh(mesh, color=color, opacity=alpha, show_edges=True)
    else:
        obj_files_2 = [f for f in os.listdir(folder_2) if f.endswith(".obj")]
        color_1 = np.random.rand(3)
        color_2 = np.random.rand(3)
        meshes_1 = append_meshes(obj_files_1 , folder_1, color_1,alpha=0.7)
        meshes_2 = append_meshes(obj_files_2 , folder_2, color_2,alpha=0.5)
        if not meshes_1:
            print("❌ 没有可用的 .obj 文件进行渲染")
            return
        plotter = pv.Plotter()
        for mesh, color, alpha in meshes_1:
            plotter.add_mesh(mesh, color=color, opacity=alpha, show_edges=True)
        for mesh, color, alpha in meshes_2:
            plotter.add_mesh(mesh, color=color, opacity=alpha, show_edges=True)
    scene_bounds = plotter.bounds  # 返回一个长度为6的元组
    print("整个场景的边界:", scene_bounds)
    plotter.add_axes()
    plotter.show(title="PyVista Transparent OBJ Viewer")



if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.withdraw()
        folder_1 = filedialog.askdirectory(title="选择第一个 .obj 文件夹（必选）")
        if not folder_1:
            messagebox.showerror("未选择文件夹", "请选择第一个有效的文件夹后再运行。")
            sys.exit(1)
        folder_2 = filedialog.askdirectory(title="选择第二个 .obj 文件夹（可选，可跳过）")
        if not folder_2:
            folder_2 = None  # 用户取消选择
        main(folder_1, folder_2)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        messagebox.showerror("程序异常", tb)
