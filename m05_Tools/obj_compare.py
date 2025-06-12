import os
import sys

import open3d as o3d
import tkinter as tk
from tkinter import filedialog, messagebox

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

def main(folder):
    obj_files = [f for f in os.listdir(folder) if f.endswith(".obj")]
    meshes = append_meshes(obj_files,folder)
    if meshes:
        o3d.visualization.draw_geometries(meshes)
    else:
        print("没有可用的 .obj 文件进行渲染")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.withdraw()
        # Ask user to select folder
        folder = filedialog.askdirectory(title="Select the folder")
        if not folder:
            messagebox.showerror("No Folder Selected", "Please select a valid folder. Exiting.")
            sys.exit(1)
        main(folder)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        messagebox.showerror("Unhandled Exception", tb)
