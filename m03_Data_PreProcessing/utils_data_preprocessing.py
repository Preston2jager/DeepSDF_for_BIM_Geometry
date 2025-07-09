import ifcopenshell
import ifcopenshell.geom
import os
import re
import shutil
import math
import tempfile
import random

import numpy as np
from collections import deque, defaultdict

def npy_read_out(file):
    data = np.load(file, allow_pickle=True)  
    print(type(data))  
    if isinstance(data, np.ndarray):
        try:
            data = data.item()
        except Exception as e:
            print(f"取出dict失败: {e}")
    print(type(data))  # 应该变成 <class 'dict'>
    print(data.keys())  # 看字典的键

def Delete_files(folder_path, file_type):
    """
    elete all files of a type in a folder.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_type):
            target_file_path = os.path.join(folder_path, filename)
            os.remove(target_file_path)
            print(f"File Deleted: {target_file_path}")

def IFC_file_expand(filename, copies, Raw_IFC_dir, Expanded_IFC_dir):
    """
    Copy and rename IFC files.
    """
    Raw_IFC_file_path = os.path.join(Raw_IFC_dir, filename)
    File_base, File_ext = os.path.splitext(filename)
    for i in range(1, copies + 1):
        Target_file_name = f"{File_base}_copy{i}{File_ext}"
        Target_file_path = os.path.join(Expanded_IFC_dir, Target_file_name)
        shutil.copy(Raw_IFC_file_path, Target_file_path)
        print(f"IFC Expansion: {Target_file_path} Created")

def Regenerate_global_ids(filename, IFC_dir):
    """
    Generate new uids for all IFC elements.
    """
    IFC_file_path = os.path.join(IFC_dir,filename)
    IFC_file = ifcopenshell.open(IFC_file_path)
    for entity in IFC_file.by_type("IfcRoot"):
        entity.GlobalId = ifcopenshell.guid.new()
    IFC_file.write(IFC_file_path) 
    print(f"New uid for {IFC_file_path} generated")

def Export_windows_and_doors(elem, idx, settings):
    """
    Replace window/door geometry with an oriented bounding box.
    """
    settings.set(settings.USE_WORLD_COORDS, False)
    vertices, triangles = [], []
    v_ofs = 0
    IFC_class = elem.is_a()
    try:
        shp  = ifcopenshell.geom.create_shape(settings, elem)
        verts_l = np.asarray(shp.geometry.verts, dtype=float).reshape(-1, 3)
        T_wl = np.asarray(shp.transformation.matrix, dtype=float).reshape(4, 4).T
        lo, hi = verts_l.min(0), verts_l.max(0)
        bbox_l = np.array([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ])
        bbox_w = (T_wl @ np.c_[bbox_l, np.ones(8)].T).T[:, :3]
        vertices.extend(bbox_w)
        quads = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4],
            [1,2,6,5], [2,3,7,6], [3,0,4,7],
        ]
        for q in quads:
            a, b, c, d = (v_ofs + np.array(q)).tolist()
            triangles += [[a, b, c], [a, c, d]]
        v_ofs += 8
    except Exception as e:
        print(f"[{elem.GlobalId}] 处理失败: {e}")
    return vertices, triangles, elem.GlobalId, IFC_class, idx, False, None

def Export_general_elements(element, index, settings):
    """
    Export verts and faces for general elements, if an elements has separate parts, export separately.
    """
    settings.set(settings.USE_WORLD_COORDS, True)
    Split = False
    IFC_class = element.is_a()
    Vertices, Triangles = Get_geometry(element, settings)
    face_graph = Build_face_graph(Triangles)
    Segments = Find_connected_components(Triangles, face_graph)
    if len(Segments) > 1:
        print(f"Element {element.GlobalId} has {len(Segments)} separate parts.")
        Split = True
    else:
        Segments = None
    return Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments

def IFC_to_obj(Objs_dir, IFC_file, IFC_classes, index, settings):
    """
    Export IFC elements to .obj files.
    """
    # Doors and windows are handled in different way.
    Special_classes = {"IfcWindow", "IfcDoor"}
    Window_and_door_elements = []
    General_elements = []
    # Extracting elements
    for class_name in IFC_classes:
        elements = IFC_file.by_type(class_name)
        if class_name in Special_classes:
            Window_and_door_elements += elements
        else:
            General_elements += elements
            print(General_elements)
    # Process windows and doors
    for element in Window_and_door_elements:
        Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments = Export_windows_and_doors(element, index, settings)
        Write_to_obj(Objs_dir, Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments)
    # Process other elements.
    for element in General_elements:
        Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments = Export_general_elements(element, index, settings)
        Write_to_obj(Objs_dir, Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments)

def Get_geometry(element, settings):
    """
    Get verts and faces of general elements.
    """
    shape = ifcopenshell.geom.create_shape(settings, element)
    verts = np.array(shape.geometry.verts).reshape(-1, 3)  # 3D 坐标
    faces = np.array(shape.geometry.faces).reshape(-1, 3)  # 三角形索引
    return verts, faces

def Build_face_graph(faces):
    """
    Generate a graph of faces to determine separate elements.
    """
    edge_to_faces = defaultdict(set)
    face_graph = defaultdict(set)
    for i, face in enumerate(faces):
        edges = {
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0]))
        }
        for edge in edges:
            edge_to_faces[edge].add(i)
    for edge, face_set in edge_to_faces.items():
        face_list = list(face_set)
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                face_graph[face_list[i]].add(face_list[j])
                face_graph[face_list[j]].add(face_list[i])
    return face_graph

def Find_connected_components(faces, face_graph):
    """
    Identify separate geometry in an element.
    """
    visited = set()
    Segements = []
    def bfs(start_face):
        queue = deque([start_face])
        component = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            queue.extend(face_graph[current] - visited)
        return component
    for i in range(len(faces)):
        if i not in visited:
            group = bfs(i)
            Segements.append(group)
    return Segements

def Write_to_obj(Objs_dir, vertices, faces, uid, ifc_class, index, split=False, groups=None):
    """
    Export IFC elements as obj, also handle elements with multiple parts.
    """
    if split:
        for split_num, group in enumerate(groups, start=1):
            used_vertex_indices = set()
            for face_index in group:
                used_vertex_indices.update(faces[face_index])
            used_vertex_indices = sorted(used_vertex_indices)
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices, start=1)}
            file_name = f"{index}_{ifc_class}_{uid}_split_{split_num}"
            output_file = os.path.join(Objs_dir, f"{file_name}.obj")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Split OBJ File\n")
                for old_idx in used_vertex_indices:
                    vertex = vertices[old_idx]
                    vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                    f.write(vertex_str)
                f.write(f"\no Object_{split_num}\n")
                for face_index in group:
                    remapped = [index_mapping[v] for v in faces[face_index]]
                    f.write("f " + " ".join(str(idx) for idx in remapped) + "\n")
            print(f"Element {uid}: Split {split_num} saved to {output_file}")
    else:
        file_name = f"{index}_{ifc_class}_{uid}.obj"
        output_file = os.path.join(Objs_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# OBJ File\n")
            for vertex in vertices:
                vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                f.write(vertex_str)
            f.write("\no Object\n")
            for face in faces:
                f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
        print(f"Element {uid}: Saved to {output_file}")

def Generate_random_transform(cfg):
    """
    Generate random rotation and translation coordinates for an OBJ file.
    """
    angle = random.uniform(cfg["rotation"]["min"], cfg["rotation"]["max"])
    tx = random.uniform(cfg["translation"]["x_range"][0], cfg["translation"]["x_range"][1])
    ty = random.uniform(cfg["translation"]["y_range"][0], cfg["translation"]["y_range"][1])
    tz = 0
    return {
        "rotation_angle": angle,
        "translation": (tx, ty, tz),
        "axis": cfg.get("axis", "y")
    }

def Rotate_vertex(vertex, angle_degrees, axis='y'):
    """
    Get rotation coords for vertices.
    """
    angle_radians = math.radians(angle_degrees)
    x, y, z = vertex
    if axis == 'y':
        x_new = x * math.cos(angle_radians) + z * math.sin(angle_radians)
        y_new = y
        z_new = -x * math.sin(angle_radians) + z * math.cos(angle_radians)
    elif axis == 'x':
        x_new = x
        y_new = y * math.cos(angle_radians) - z * math.sin(angle_radians)
        z_new = y * math.sin(angle_radians) + z * math.cos(angle_radians)
    elif axis == 'z':
        x_new = x * math.cos(angle_radians) - y * math.sin(angle_radians)
        y_new = x * math.sin(angle_radians) + y * math.cos(angle_radians)
        z_new = z
    else:
        x_new, y_new, z_new = x, y, z
    return [x_new, y_new, z_new]

def Single_obj_transform(file_path, rotation_angle=0, translation=(0, 0, 0), axis='y'):
    """
    Rotate and move a single .obj file.
    """
    vertices = []
    faces = []
    others = []  
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertex = Rotate_vertex(vertex, rotation_angle, axis)
                vertex = [vertex[i] + translation[i] for i in range(3)]
                vertices.append(vertex)
            elif line.startswith('f '):
                faces.append(line.strip())
            else:
                others.append(line.rstrip())
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        for v in vertices:
            tmp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in others:
            tmp.write(f"{line}\n")
        for face in faces:
            tmp.write(f"{face}\n")
        temp_path = tmp.name
    shutil.move(temp_path, file_path)

def Batch_obj_transform(directory, cfg):
    """
    Randomly rotate and move all expanded .obj files.
    Use prefix to identify file from same project model.
    """
    prefix_transform_map = {}
    for filename in os.listdir(directory):
        match = re.match(r'^(\d+)_.*\.obj$', filename, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            if prefix not in prefix_transform_map:
                prefix_transform_map[prefix] = Generate_random_transform(cfg)
            params = prefix_transform_map[prefix]
            file_path = os.path.join(directory, filename)
            Single_obj_transform(
                file_path,
                rotation_angle=params["rotation_angle"],
                translation=params["translation"],
                axis=params["axis"]
            )
            print(f"{filename} Transformed")
        else:
            print(f"[!] Skipped {filename}")