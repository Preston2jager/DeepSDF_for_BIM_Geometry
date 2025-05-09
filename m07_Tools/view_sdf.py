import numpy as np
import open3d as o3d
import os
import data.converted_data
from plyer import notification


# 加载数据
file = os.path.join(os.path.dirname(data.converted_data.__file__), 'samples_dict.npy')
samples_dict = np.load(file, allow_pickle=True).item()

# 设置颜色：正数-红，负数-蓝，零-绿
colors = {
    'positive': [1, 0, 0, 0.5],
    'negative': [0, 0, 1, 0.5],
    'zero':     [0, 1, 0, 0.5]
}

# 准备点云列表
points_all = []
colors_all = []

# 选择一个对象可视化（比如第一个）
obj_data = samples_dict[0]  # 或换成你想查看的 index

points = obj_data['samples_latent_class'][:, 1:]  # 去除 latent code


sdf = obj_data['sdf']

# 设定一个非常小的阈值判断“接近0”
epsilon = 1e-5
points_real = []
# 按 SDF 分类并赋颜色
for i in range(points.shape[0]):
    p = points[i]
    d = sdf[i]
    if d > epsilon:
        c = colors['positive'][:3]
    elif d < -epsilon:
        c = colors['negative'][:3]
    else:
        c = colors['zero'][:3]
        points_real.append(p)
    points_all.append(p)
    colors_all.append(c)

min_coords = np.min(points_real, axis=0)  # 得到 x, y, z 方向的最小值
max_coords = np.max(points_real, axis=0)  # 得到 x, y, z 方向的最大值
dimensions = max_coords - min_coords  # 分别计算 x, y, z 方向的长度
print("Dimensions (width, height, depth):", dimensions)

# 转为 numpy 数组
points_all = np.array(points_all)
colors_all = np.array(colors_all)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_all)
pcd.colors = o3d.utility.Vector3dVector(colors_all)  # Open3D 不支持 alpha，但我们保留 RGB

notification.notify(
    title='View',
    message='Start Rending',
    timeout=5  # 秒
)

# 可视化
o3d.visualization.draw_geometries([pcd])
