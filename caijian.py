import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud('raw_pcd_data/frame_0100.pcd')

# 1. 强制裁剪一个非常小的中心区域 (假设鸭子在相机正前方 60cm 处)
# 我们只取 X: -10cm到10cm, Y: -10cm到10cm, Z: 50cm到90cm
min_bound = np.array([-0.1, -0.1, 0.5])
max_bound = np.array([0.1, 0.1, 0.9])
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
duck_candidate = pcd.crop(bbox)

# 2. 如果没切到，扩大一点点范围再试
if duck_candidate.is_empty():
    print("中心区域没找到，正在扩大搜索范围...")
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-0.3, -0.3, 0.4]), np.array([0.3, 0.3, 1.2]))
    duck_candidate = pcd.crop(bbox)

# 3. 核心：移除背景平面 (RANSAC)
# 这行代码会识别出点云里最大的平面（桌子），然后把它删掉，剩下的就是鸭子
plane_model, inliers = duck_candidate.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
duck_only = duck_candidate.select_by_index(inliers, invert=True)

# 4. 可视化：开启光照和黑色背景
print("显示结果：请在窗口中按 'L' 开启光照，按 '/' 切换黑底")
o3d.visualization.draw_geometries([duck_only])