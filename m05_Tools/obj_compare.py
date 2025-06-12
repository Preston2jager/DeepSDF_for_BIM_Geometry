import os
import sys
import numpy as np
import pyvista as pv
import tkinter as tk
from tkinter import filedialog, messagebox

def append_meshes(obj_files, obj_path, alpha=0.5):
    """
    加载 obj 文件并赋予随机颜色和透明度。
    """
    meshes = []
    for obj_file in obj_files:
        full_path = os.path.join(obj_path, obj_file)
        try:
            mesh = pv.read(full_path)
            color = np.random.rand(3)
            mesh["colors"] = np.tile(color, (mesh.n_points, 1))
            meshes.append((mesh, color, alpha))
        except Exception as e:
            print(f"⚠️ 无法加载 {obj_file}: {e}")
    return meshes

def main(folder):
    obj_files = [f for f in os.listdir(folder) if f.endswith(".obj")]
    meshes = append_meshes(obj_files, folder, alpha=0.5)

    if not meshes:
        print("❌ 没有可用的 .obj 文件进行渲染")
        return

    plotter = pv.Plotter()
    for mesh, color, alpha in meshes:
        plotter.add_mesh(mesh, color=color, opacity=alpha, show_edges=True)

    plotter.add_axes()
    plotter.show(title="PyVista Transparent OBJ Viewer")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="选择包含 .obj 的文件夹")
        if not folder:
            messagebox.showerror("未选择文件夹", "请选择有效的文件夹后再运行。")
            sys.exit(1)
        main(folder)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        messagebox.showerror("程序异常", tb)
