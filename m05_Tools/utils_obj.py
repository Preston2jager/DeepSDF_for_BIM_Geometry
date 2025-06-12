import os, re, math, random, tempfile, shutil, yaml

import config_files
import data.objects

def rotate_vertex(vertex, angle_degrees, axis='y'):
    angle_radians = math.radians(angle_degrees)
    x, y, z = vertex
    if axis == 'y':
        # 绕 y 轴旋转：保持 y 不变
        x_new = x * math.cos(angle_radians) + z * math.sin(angle_radians)
        y_new = y
        z_new = -x * math.sin(angle_radians) + z * math.cos(angle_radians)
    elif axis == 'x':
        # 绕 x 轴旋转：保持 x 不变
        x_new = x
        y_new = y * math.cos(angle_radians) - z * math.sin(angle_radians)
        z_new = y * math.sin(angle_radians) + z * math.cos(angle_radians)
    elif axis == 'z':
        # 绕 z 轴旋转：保持 z 不变
        x_new = x * math.cos(angle_radians) - y * math.sin(angle_radians)
        y_new = x * math.sin(angle_radians) + y * math.cos(angle_radians)
        z_new = z
    else:
        x_new, y_new, z_new = x, y, z
    return [x_new, y_new, z_new]

def transform_obj_file(file_path, rotation_angle=0, translation=(0, 0, 0), axis='y'):
    vertices = []
    faces = []
    others = []  # 保留其它信息，如 vn, vt, o, g 等

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                # 旋转
                vertex = rotate_vertex(vertex, rotation_angle, axis)
                # 平移（仅对 x 和 y 平移，z 轴保持不变）
                vertex = [vertex[i] + translation[i] for i in range(3)]
                vertices.append(vertex)
            elif line.startswith('f '):
                faces.append(line.strip())
            else:
                others.append(line.rstrip())

    # 使用临时文件写入，保证过程安全，再替换原文件
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        for v in vertices:
            tmp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in others:
            tmp.write(f"{line}\n")
        for face in faces:
            tmp.write(f"{face}\n")
        temp_path = tmp.name

    shutil.move(temp_path, file_path)

def generate_random_transform(cfg):
    # 生成指定范围内的随机旋转角度
    angle = random.uniform(cfg["rotation"]["min"], cfg["rotation"]["max"])
    # 随机生成平移向量：仅在水平面内（x, y），z 保持为 0
    tx = random.uniform(cfg["translation"]["x_range"][0], cfg["translation"]["x_range"][1])
    ty = random.uniform(cfg["translation"]["y_range"][0], cfg["translation"]["y_range"][1])
    tz = 0
    return {
        "rotation_angle": angle,
        "translation": (tx, ty, tz),
        "axis": cfg.get("axis", "y")
    }

def batch_transform_by_prefix(directory, cfg):
    # 存储每个数字前缀对应的随机变换参数，确保同一前缀的文件使用同一组参数
    prefix_transform_map = {}

    for filename in os.listdir(directory):
        # 匹配以数字开头，后跟任意字符，并以 .obj 结尾的文件
        match = re.match(r'^(\d+)_.*\.obj$', filename, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            # 如果该前缀还未生成变换参数，则生成一组随机参数
            if prefix not in prefix_transform_map:
                prefix_transform_map[prefix] = generate_random_transform(cfg)
            params = prefix_transform_map[prefix]
            file_path = os.path.join(directory, filename)
            transform_obj_file(
                file_path,
                rotation_angle=params["rotation_angle"],
                translation=params["translation"],
                axis=params["axis"]
            )
            print(f"[✓] Transformed {filename} with prefix '{prefix}': {params}")
        else:
            print(f"[!] Skipped {filename} (不符合匹配规则)")

def main(cfg):
    # 批量处理当前目录下的 .obj 文件
    objs_dir = os.path.dirname(data.objects.__file__)
    batch_transform_by_prefix(objs_dir, cfg)

if __name__=='__main__':
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'extract_ifc.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)