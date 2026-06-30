#!/usr/bin/env python3
"""
Duck 3D Point Cloud Extraction Pipeline - 完全修复版 v3
核心修复：
  1. 彻底放弃全场景ICP，改用 Duck-to-Duck ICP
     （鸭子有曲面+喙，ICP约束充分；桌面是大平面，ICP无约束）
  2. 每帧对变换幅度做物理约束（平移<4cm/帧，旋转<6°/帧）
  3. 固定关键帧策略，防止滚动累积漂移
  4. DBSCAN最终聚类，强制保留最大连通块（鸭子主体）
  5. 保留所有原有功能（分割/滤波/RANSAC）
"""

import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d
import numpy as np
import open3d as o3d
from collections import deque

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError:
    print("[ERROR] 需要ROS2环境。")
    sys.exit(1)

SAM2_AVAILABLE = False
try:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
    print("[INFO] SAM2 已找到")
except ImportError:
    print("[ERROR] SAM2未安装")
    sys.exit(1)


# ══════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)


# ══════════════════════════════════════════════════════
# 1. Bag读取器（原版不变）
# ══════════════════════════════════════════════════════
class BagReader:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        cr = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(sr, cr)
        self.type_map = {t.name: t.type
                         for t in self.reader.get_all_topics_and_types()}

    def read_all(self, max_frames=None):
        color_buf, points_buf = deque(), deque()
        paired = []
        while self.reader.has_next():
            topic, raw, ts = self.reader.read_next()
            if topic == "/camera/color/image_raw":
                msg = deserialize_message(raw, get_message(self.type_map[topic]))
                img = self._decode_image(msg)
                if img is not None:
                    color_buf.append((ts, img))
            elif topic == "/camera/depth/points":
                msg = deserialize_message(raw, get_message(self.type_map[topic]))
                pts = self._decode_pointcloud2(msg)
                if pts is not None:
                    points_buf.append((ts, pts))

            while color_buf and points_buf:
                tc, ic = color_buf[0]
                tp, pp = points_buf[0]
                dt = abs(tc - tp) / 1e6
                if dt < 33.0:
                    paired.append((ic, pp, tc))
                    color_buf.popleft()
                    points_buf.popleft()
                    if max_frames and len(paired) >= max_frames:
                        print(f"[BAG] 达到 {max_frames} 帧")
                        return paired
                elif tc < tp:
                    color_buf.popleft()
                else:
                    points_buf.popleft()

        print(f"[BAG] 共配对 {len(paired)} 帧")
        return paired

    @staticmethod
    def _decode_image(msg):
        enc = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        h, w = msg.height, msg.width
        if enc in ("rgb8", "rgb"):
            return data.reshape(h, w, 3)
        elif enc in ("bgr8", "bgr"):
            return data.reshape(h, w, 3)[:, :, ::-1].copy()
        return None

    @staticmethod
    def _decode_pointcloud2(msg):
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")):
            return None
        step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        n = msg.width * msg.height
        result = np.zeros((n, 6), dtype=np.float32)
        for i, name in enumerate(["x", "y", "z"]):
            result[:, i] = _extract_field_fast(
                data, fields[name].offset, step, n, np.float32)
        if "rgb" in fields:
            raw = _extract_field_fast(
                data, fields["rgb"].offset, step, n, np.float32)
            rgb_int = raw.view(np.uint32)
            result[:, 3] = ((rgb_int >> 16) & 0xFF).astype(np.float32)
            result[:, 4] = ((rgb_int >> 8) & 0xFF).astype(np.float32)
            result[:, 5] = (rgb_int & 0xFF).astype(np.float32)
        else:
            result[:, 3:] = 200.0
        valid = np.isfinite(result[:, :3]).all(axis=1) & (result[:, 2] > 0.01)
        return result[valid]


# ══════════════════════════════════════════════════════
# 2. 相机参数读取（原版不变）
# ══════════════════════════════════════════════════════
def get_color_intrinsics(bag_path):
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/camera/color/camera_info":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k
            print(f"[INTR] 彩色相机内参: fx={K[0]:.2f}, fy={K[4]:.2f}, "
                  f"cx={K[2]:.2f}, cy={K[5]:.2f}, 分辨率={msg.width}x{msg.height}")
            return (K[0], K[4], K[2], K[5])
    print("[WARN] 未找到彩色内参，使用默认值")
    return (691.33, 691.51, 643.92, 362.12)


def get_extrinsics_depth_to_color(bag_path):
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    adj = {}

    def quat_to_mat(tx, ty, tz, qx, qy, qz, qw):
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = 1 - 2*qy**2 - 2*qz**2
        mat[0, 1] = 2*qx*qy - 2*qz*qw
        mat[0, 2] = 2*qx*qz + 2*qy*qw
        mat[1, 0] = 2*qx*qy + 2*qz*qw
        mat[1, 1] = 1 - 2*qx**2 - 2*qz**2
        mat[1, 2] = 2*qy*qz - 2*qx*qw
        mat[2, 0] = 2*qx*qz - 2*qy*qw
        mat[2, 1] = 2*qy*qz + 2*qx*qw
        mat[2, 2] = 1 - 2*qx**2 - 2*qy**2
        mat[0, 3], mat[1, 3], mat[2, 3] = tx, ty, tz
        return mat

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/tf_static":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            for tf in msg.transforms:
                p = tf.header.frame_id
                c = tf.child_frame_id
                t, r = tf.transform.translation, tf.transform.rotation
                mat = quat_to_mat(t.x, t.y, t.z, r.x, r.y, r.z, r.w)
                adj.setdefault(p, []).append((c, mat))
                adj.setdefault(c, []).append((p, np.linalg.inv(mat)))
            break

    src = "camera_depth_optical_frame"
    dst = "camera_color_optical_frame"
    queue = deque([(src, np.eye(4, dtype=np.float64))])
    visited = {src}
    while queue:
        curr, mat = queue.popleft()
        if curr == dst:
            print(f"[TF] 找到外参矩阵 {src} → {dst}")
            print(f"[TF] 平移部分(mm): x={mat[0,3]*1000:.1f}, "
                  f"y={mat[1,3]*1000:.1f}, z={mat[2,3]*1000:.1f}")
            return mat
        for nxt, step_mat in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, step_mat @ mat))

    print("[WARN] 未找到TF外参，使用单位矩阵")
    return np.eye(4, dtype=np.float64)


# ══════════════════════════════════════════════════════
# 3. SAM2分割器（原版不变）
# ══════════════════════════════════════════════════════
class DuckSegmenter:
    def __init__(self, sam2_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SEG] 加载SAM2，设备: {device}")
        if "tiny" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "small" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "base_plus" in sam2_checkpoint:
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(
            cfg, sam2_checkpoint, device=device)

    def get_clicks(self, rgb_image):
        h, w = rgb_image.shape[:2]
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        scale = min(1.0, 1280/w, 720/h)
        disp = cv2.resize(display, (int(w*scale), int(h*scale)))
        pos, neg, all_pts = [], [], []

        print("\n左键=正样本(鸭子) | 右键=负样本(桌面/背景) | ENTER=确认 | Z=撤销")
        canvas = [disp.copy()]

        def redraw():
            img = canvas[0].copy()
            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,255,0), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255,255,255), 2)
            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,0,255), -1)
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 9, (255,255,255), 2)
            cv2.putText(img, f"绿(正):{len(pos)} 红(负):{len(neg)} [ENTER确认]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            return img

        def on_mouse(event, x, y, flags, param):
            rx, ry = x/scale, y/scale
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([rx, ry]); all_pts.append((rx, ry, True))
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([rx, ry]); all_pts.append((rx, ry, False))

        cv2.namedWindow("Duck_Annotation", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Duck_Annotation", on_mouse)
        while True:
            cv2.imshow("Duck_Annotation", redraw())
            key = cv2.waitKey(30) & 0xFF
            if key in [13, 32] and len(pos) > 0:
                break
            elif key in [ord('z'), ord('Z')] and all_pts:
                _, _, is_pos = all_pts.pop()
                pos.pop() if is_pos else neg.pop()
        cv2.destroyAllWindows()
        return np.array(pos), np.array(neg)

    def segment_all(self, frames_rgb, pos_pts, neg_pts):
        import tempfile, shutil
        from PIL import Image
        tmp = tempfile.mkdtemp(prefix="sam2_")
        try:
            for i, rgb in enumerate(frames_rgb):
                Image.fromarray(rgb).save(os.path.join(tmp, f"{i:05d}.jpg"),
                                          quality=95)
            with torch.inference_mode():
                state = self.predictor.init_state(video_path=tmp)
                pts = np.vstack([pos_pts, neg_pts]).astype(np.float32)
                lbs = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=pts, labels=lbs)
                masks = [None] * len(frames_rgb)
                for idx, _, logits in self.predictor.propagate_in_video(state):
                    masks[idx] = (logits[0, 0] > 0.0).cpu().numpy()
                    if idx % 100 == 0:
                        cov = masks[idx].sum() / masks[idx].size * 100
                        print(f"[SAM2] 帧{idx}: mask覆盖{cov:.1f}%")
            h, w = frames_rgb[0].shape[:2]
            return [m if m is not None else np.zeros((h, w), bool)
                    for m in masks]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════
# 4. 点云过滤（原版不变）
# ══════════════════════════════════════════════════════
def filter_duck_points(pcd_nx6, rgb_image, mask,
                       color_intrinsics, extrinsics_d2c):
    fx, fy, cx, cy = color_intrinsics
    pts = pcd_nx6[:, :3].astype(np.float64)
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])
    pts_color = (extrinsics_d2c @ pts_h.T).T
    Xc, Yc, Zc = pts_color[:, 0], pts_color[:, 1], pts_color[:, 2]
    valid_z = Zc > 0.05
    u = np.zeros(len(pcd_nx6), dtype=np.int32)
    v = np.zeros(len(pcd_nx6), dtype=np.int32)
    u[valid_z] = (Xc[valid_z] / Zc[valid_z] * fx + cx).astype(np.int32)
    v[valid_z] = (Yc[valid_z] / Zc[valid_z] * fy + cy).astype(np.int32)
    H, W = mask.shape
    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    in_mask = np.zeros(len(pcd_nx6), dtype=bool)
    idx = np.where(in_bounds)[0]
    in_mask[idx] = mask[v[idx], u[idx]]
    result = pcd_nx6[in_mask].copy()
    if result.shape[1] >= 6:
        result[:, 3:6] = rgb_image[v[in_mask], u[in_mask]].astype(np.float32)
    return result


def remove_floor_ransac(pcd_nx6, dist_thresh=0.012):
    if len(pcd_nx6) < 50:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    try:
        plane_model, inliers = o3pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=500)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        inlier_ratio = len(inliers) / len(pcd_nx6)
        is_floor = (abs(normal[1]) > 0.6 or abs(normal[2]) > 0.6)
        if is_floor and inlier_ratio > 0.15:
            print(f"[RANSAC] 检测到地面，法向量={normal.round(3)}，"
                  f"占比={inlier_ratio:.1%}，删除")
            keep = np.ones(len(pcd_nx6), dtype=bool)
            keep[inliers] = False
            return pcd_nx6[keep]
        else:
            print(f"[RANSAC] 未检测到明显地面，保留")
    except Exception as e:
        print(f"[RANSAC] 失败: {e}")
    return pcd_nx6


def statistical_filter(pcd_nx6, nb=20, std=1.5):
    if len(pcd_nx6) < nb + 1:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    _, ind = o3pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    print(f"[FILTER] 统计滤波: {len(pcd_nx6)} → {len(ind)} 点")
    return pcd_nx6[ind]


def voxel_down(pcd_nx6, voxel_size):
    if len(pcd_nx6) < 10:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    if pcd_nx6.shape[1] >= 6:
        o3pcd.colors = o3d.utility.Vector3dVector(
            np.clip(pcd_nx6[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    down = o3pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(down.points, dtype=np.float32)
    if down.has_colors():
        cols = (np.asarray(down.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts


# ══════════════════════════════════════════════════════
# 5. 核心修复：Duck-to-Duck ICP + 变换校验 + DBSCAN清理
# ══════════════════════════════════════════════════════
def _nx6_to_pcd(nx6, voxel_size):
    if nx6 is None or len(nx6) < 30:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nx6[:, :3].astype(np.float64))
    if nx6.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(nx6[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) < 20:
        return None
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    return pcd
def _centroid_init(src, tgt):
    """
    用质心差估计初始平移。
    适合鸭子这种近似凸体：不同视角的质心差约等于相机偏移量的反方向。
    当FPFH歧义大时，这个粗估计能提供更好的初始化。
    """
    src_c = np.asarray(src.points).mean(axis=0)
    tgt_c = np.asarray(tgt.points).mean(axis=0)
    tf = np.eye(4)
    tf[:3, 3] = tgt_c - src_c
    return tf
 
 
def _registration_src_to_global(src_pcd, global_pcd, voxel_size):
    """
    把 src_pcd 配准到 global_pcd（全局累积点云）。
    
    步骤：
    1. 质心对齐，得到粗初始平移
    2. FPFH + RANSAC，从质心初始化开始搜索（减少镜像歧义）
    3. ICP精配准
    
    返回：(变换矩阵, fitness)
    """
    fpfh_voxel = max(voxel_size * 8, 0.008)   # 比单帧配准时更大一点
 
    # 降采样用于FPFH
    src_ds = src_pcd.voxel_down_sample(fpfh_voxel)
    tgt_ds = global_pcd.voxel_down_sample(fpfh_voxel)
 
    if len(src_ds.points) < 10 or len(tgt_ds.points) < 10:
        return np.eye(4), 0.0, "less_point"
 
    for pcd in [src_ds, tgt_ds]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=fpfh_voxel * 3, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
 
    def fpfh(pcd, vox):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(
                radius=vox * 5, max_nn=100))
 
    src_feat = fpfh(src_ds, fpfh_voxel)
    tgt_feat = fpfh(tgt_ds, fpfh_voxel)
 
    dist_coarse = fpfh_voxel * 2.0
 
    # 先用质心初始化做一次ICP，得到一个基准fitness
    init_tf = _centroid_init(src_pcd, global_pcd)
    result_centroid = o3d.pipelines.registration.registration_icp(
        src_pcd, global_pcd,
        max_correspondence_distance=voxel_size * 3,
        init=init_tf,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50))
 
    # FPFH + RANSAC全局搜索
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds, src_feat, tgt_feat,
        mutual_filter=False,   # 关掉mutual filter！这是"Too few correspondences"警告的来源
        max_correspondence_distance=dist_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_coarse),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=200000, confidence=0.999))
 
    # ICP精配准（从RANSAC结果出发）
    result_icp_ransac = o3d.pipelines.registration.registration_icp(
        src_pcd, global_pcd,
        max_correspondence_distance=voxel_size * 2,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100))
 
    # 选择更好的结果（质心初始化 vs FPFH）
    if result_centroid.fitness >= result_icp_ransac.fitness:
        best_tf = result_centroid.transformation
        best_fitness = result_centroid.fitness
        method = "质心+ICP"
    else:
        best_tf = result_icp_ransac.transformation
        best_fitness = result_icp_ransac.fitness
        method = "FPFH+ICP"
 
    return best_tf, best_fitness, method
 
 
def _check_transform_z(tf, max_z_shift=0.05):
    """
    只检查Z方向位移（高度）。
    鸭子在桌上不会飞起来，Z方向移动超过5cm必然是配准错误。
    XY方向（水平）不限制，因为相机确实在绕鸭子转。
    """
    dz = abs(tf[2, 3])
    cos_a = np.clip((np.trace(tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(abs(cos_a))))
    ok = dz < max_z_shift
    return ok, dz, angle_deg
 
 
def merge_with_global_target(duck_arrays,
                              voxel_size_duck=0.001,
                              max_z_shift=0.05,
                              min_fitness=0.40,
                              batch_size=5):
    """
    新策略：每个src帧直接配准到累积全局点云
    
    参数：
      voxel_size_duck: 最终点云体素大小（m）
      max_z_shift: 每帧允许的最大高度变化（m），默认5cm
      min_fitness: ICP配准接受阈值
      batch_size: 每batch_size帧做一次全局点云更新（太频繁会慢）
    
    运行命令示例：
      python duck_sam2_FPFH_RANSAC_icp.py \
        --bag xxx.db3 \
        --sam2_checkpoint sam2.1_hiera_tiny.pt \
        --output duck_v5.pcd \
        --use_global_target       ← 新增flag，调用这个函数
        --voxel_size 0.001
    """
    if not duck_arrays:
        return np.array([])
 
    n = len(duck_arrays)
    icp_voxel = max(voxel_size_duck * 4, 0.004)
 
    print(f"\n[REG] 全局累积点云配准，共 {n} 帧")
    print(f"[REG] voxel={icp_voxel:.4f}m, min_fitness={min_fitness}, "
          f"max_z_shift={max_z_shift*100:.1f}cm")
 
    # 初始化：第0帧作为全局点云起点
    pcd0 = _nx6_to_pcd(duck_arrays[0], icp_voxel)
    if pcd0 is None:
        print("[REG] 第0帧无效")
        return np.array([])
 
    global_pcd = pcd0
 
    # 全局点云对应的nx6数组（用于最终输出）
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        duck_arrays[0][:, :3].astype(np.float64))
    if duck_arrays[0].shape[1] >= 6:
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(duck_arrays[0][:, 3:6] / 255.0, 0, 1))
 
    accepted = 0
    rejected = 0
    pending_for_global = []   # 等待加入全局点云的已接受帧
 
    for i in range(1, n):
        src_raw = duck_arrays[i]
        src_pcd = _nx6_to_pcd(src_raw, icp_voxel)
        if src_pcd is None:
            rejected += 1
            continue
 
        # 配准到全局点云
        result = _registration_src_to_global(src_pcd, global_pcd, icp_voxel)
        tf, fitness, method = result
 
        accepted_this = False
 
        if fitness < min_fitness:
            rejected += 1
            if i % 20 == 0:
                print(f"[REG] 帧{i:3d}: fitness={fitness:.3f} 过低({method}) → 拒绝")
        else:
            ok, dz, ang = _check_transform_z(tf, max_z_shift)
            if not ok:
                rejected += 1
                if i % 20 == 0:
                    print(f"[REG] 帧{i:3d}: fitness={fitness:.3f} "
                          f"Δz={dz*100:.1f}cm({method}) 超限 → 拒绝")
            else:
                accepted_this = True
                accepted += 1
 
                curr = o3d.geometry.PointCloud()
                curr.points = o3d.utility.Vector3dVector(
                    src_raw[:, :3].astype(np.float64))
                if src_raw.shape[1] >= 6:
                    curr.colors = o3d.utility.Vector3dVector(
                        np.clip(src_raw[:, 3:6] / 255.0, 0, 1))
                curr.transform(tf)
 
                merged += curr
                merged = merged.voxel_down_sample(voxel_size_duck)
 
                pending_for_global.append(curr)
 
                if i % 10 == 0:
                    print(f"[REG] 帧{i:3d}/{n}: fitness={fitness:.3f}, "
                          f"Δz={dz*100:.1f}cm, Δθ={ang:.1f}°, "
                          f"method={method}, 接受={accepted}, 点云={len(merged.points)}")
 
        # 每batch_size帧把接受的帧合并进全局点云
        if len(pending_for_global) >= batch_size:
            for p in pending_for_global:
                global_pcd += p
            global_pcd = global_pcd.voxel_down_sample(icp_voxel)
            global_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=icp_voxel * 4, max_nn=30))
            global_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
            pending_for_global = []
            print(f"[REG] 帧{i:3d}: 全局点云更新 → {len(global_pcd.points)}点")
 
    print(f"\n[REG] 完成: 接受={accepted}/{n-1} ({accepted/(n-1)*100:.1f}%), "
          f"点云={len(merged.points)}点")
 
    pts = np.asarray(merged.points, dtype=np.float32)
    if merged.has_colors():
        cols = (np.asarray(merged.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts

def _check_transform(tf, max_trans_m, max_angle_deg):
    trans_m = float(np.linalg.norm(tf[:3, 3]))
    cos_a = np.clip((np.trace(tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(abs(cos_a))))
    ok = (trans_m < max_trans_m) and (angle_deg < max_angle_deg)
    return ok, trans_m, angle_deg



def _duck_registration(src, tgt, voxel_size):
    """
    FPFH 粗配准 → Point-to-Plane ICP 精配准
    
    为什么旧代码 fitness=0.000：
      Point-to-Plane ICP 只能收敛到 ±15° 以内的初始偏差。
      相机绕鸭子转一圈，相邻关键帧之间可能差 30-60°，ICP 直接发散。
    
    FPFH（Fast Point Feature Histograms）的优势：
      - 基于法向量方向分布做描述子，不依赖初始位姿
      - 只要点云有足够的几何特征（鸭头球面/身体曲面/喙）就能匹配
      - 鸭子恰好有丰富的法向量变化，FPFH 效果很好
    """
    # ── 1. 粗配准用更大的 voxel（特征点更稳定）──
    fpfh_voxel = voxel_size * 5   # 通常 0.005~0.010m
 
    src_ds = src.voxel_down_sample(fpfh_voxel)
    tgt_ds = tgt.voxel_down_sample(fpfh_voxel)
 
    # 重新估计法向量（fpfh_voxel 尺度）
    for pcd in [src_ds, tgt_ds]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=fpfh_voxel * 3, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
 
    # 计算 FPFH 特征
    def compute_fpfh(pcd, vox):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=vox * 5, max_nn=100))
 
    src_feat = compute_fpfh(src_ds, fpfh_voxel)
    tgt_feat = compute_fpfh(tgt_ds, fpfh_voxel)
 
    # RANSAC 粗配准
    dist_thresh_coarse = fpfh_voxel * 1.5
    result_coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds, src_feat, tgt_feat,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh_coarse),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000, confidence=0.999))
 
    # 粗配准 fitness 太差就放弃（避免垃圾初始化送进 ICP）
    if result_coarse.fitness < 0.15:
        return np.eye(4), 0.0
 
    # ── 2. ICP 精配准（有了好的初始位姿，收敛有保障）──
    dist_thresh_fine = voxel_size * 2.0
    result_fine = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=dist_thresh_fine,
        init=result_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6))
 
    return result_fine.transformation, result_fine.fitness



def _duck_icp(src, tgt, voxel_size):
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * 4,
        init=np.eye(4),
        estimation_method=
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6))
    return result.transformation, result.fitness


def keep_largest_cluster(pcd_nx6, eps=0.03, min_pts=10):
    """
    原版 eps=0.05 在点云密度不均匀时会把鸭子切成多段。
    改为 eps=0.03，更细粒度的聚类，但也意味着鸭子各部位需要连续。
    
    如果 DBSCAN 后点数<60%，说明 ICP 还有帧没对齐，不是 DBSCAN 的问题。
    """
    if len(pcd_nx6) < min_pts * 2:
        return pcd_nx6
    o3pcd = o3d.geometry.PointCloud()
    o3pcd.points = o3d.utility.Vector3dVector(pcd_nx6[:, :3].astype(np.float64))
    labels = np.array(o3pcd.cluster_dbscan(
        eps=eps, min_points=min_pts, print_progress=False))
    if labels.max() < 0:
        print("[DBSCAN] 未找到任何聚类，返回原始点云")
        return pcd_nx6
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    mask = (labels == largest_label)
    kept = pcd_nx6[mask]
    ratio = len(kept) / len(pcd_nx6)
    print(f"[DBSCAN] {len(pcd_nx6)} → {len(kept)} 点 "
          f"（共 {len(unique)} cluster，最大占比 {ratio:.1%}）")
    if ratio < 0.60:
        print(f"[WARN] DBSCAN 只保留了 {ratio:.1%}，"
              f"说明 ICP 仍有帧未对齐，点云碎片化严重。")
        print(f"  建议：检查 masks_preview/ 中 fitness=0 的帧，"
              f"确认 SAM2 mask 是否覆盖了鸭子。")
    return kept


def merge_duck_icp(duck_arrays,
                   voxel_size_duck=0.001,
                   max_trans_m=0.35,
                   max_angle_deg=50.0,
                   keyframe_interval=3):
    """
    参数说明（对应命令行参数）：
      --max_trans 0.35   手持绕行，每关键帧间隔最多移动 35cm，很宽松
      --max_angle 50.0   每关键帧间隔最多旋转 50°，对应 kf_interval=3 每帧约转 17°
      --kf_interval 3    每 3 帧更新关键帧，视频 ~30fps 时相当于 100ms
      
    如果仍有大量拒绝，考虑增大 kf_interval（更频繁更新关键帧）
    或适当放宽 max_angle（比如改到 60°）
    """
    if not duck_arrays:
        return np.array([])
 
    n = len(duck_arrays)
    print(f"\n[ICP] 开始 FPFH+ICP 配准，共 {n} 帧")
    print(f"[ICP] 参数: max_trans={max_trans_m*100:.1f}cm, "
          f"max_angle={max_angle_deg:.1f}°, "
          f"keyframe_interval={keyframe_interval}")
 
    icp_voxel = max(voxel_size_duck * 4, 0.004)
 
    kf_idx = 0
    kf_raw = duck_arrays[0]
    kf_pcd = _nx6_to_pcd(kf_raw, icp_voxel)
    if kf_pcd is None:
        print("[ICP] 第0帧无效")
        return np.array([])
 
    kf_world_tf = np.eye(4)
 
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(kf_raw[:, :3].astype(np.float64))
    if kf_raw.shape[1] >= 6:
        merged.colors = o3d.utility.Vector3dVector(
            np.clip(kf_raw[:, 3:6] / 255.0, 0, 1))
 
    accepted = 0
    rejected = 0
 
    for i in range(1, n):
        src_raw = duck_arrays[i]
        src_pcd = _nx6_to_pcd(src_raw, icp_voxel)
        if src_pcd is None or len(src_pcd.points) < 20:
            rejected += 1
            continue
 
        # 两步配准（粗 + 精）
        tf, fitness = _duck_registration(src_pcd, kf_pcd, icp_voxel)
 
        accepted_this = False
 
        if fitness < 0.35:
            print(f"[ICP] 帧{i:3d}: fitness={fitness:.3f} 过低 → 拒绝")
            rejected += 1
        else:
            ok, t_m, ang = _check_transform(tf, max_trans_m, max_angle_deg)
            if not ok:
                print(f"[ICP] 帧{i:3d}: fitness={fitness:.3f} "
                      f"Δt={t_m*100:.1f}cm Δθ={ang:.1f}° 超限 → 拒绝")
                rejected += 1
            else:
                accepted_this = True
                accepted += 1
 
                world_tf = kf_world_tf @ tf
 
                curr = o3d.geometry.PointCloud()
                curr.points = o3d.utility.Vector3dVector(
                    src_raw[:, :3].astype(np.float64))
                if src_raw.shape[1] >= 6:
                    curr.colors = o3d.utility.Vector3dVector(
                        np.clip(src_raw[:, 3:6] / 255.0, 0, 1))
                curr.transform(world_tf)
 
                merged += curr
                merged = merged.voxel_down_sample(voxel_size_duck)
 
                if i % 10 == 0:
                    print(f"[ICP] 帧{i:3d}/{n}: fitness={fitness:.3f}, "
                          f"Δt={t_m*100:.1f}cm, Δθ={ang:.1f}°, "
                          f"已接受={accepted}, 点云={len(merged.points)}")
 
        # 关键帧更新（逻辑不变）
        if (i - kf_idx) >= keyframe_interval:
            if accepted_this:
                new_kf = _nx6_to_pcd(src_raw, icp_voxel)
                if new_kf is not None:
                    kf_pcd = new_kf
                    kf_world_tf = world_tf.copy()
                    kf_idx = i
                    print(f"[ICP] 帧{i:3d}: ✓ 关键帧更新")
            else:
                kf_idx = i
                print(f"[ICP] 帧{i:3d}: ✗ 关键帧时间到但被拒，重置计数")
 
    print(f"\n[ICP] 完成: 接受={accepted}/{n-1} 帧 "
          f"({accepted/(n-1)*100:.1f}%), "
          f"最终点云={len(merged.points)} 点")
 
    if accepted / (n - 1) < 0.5:
        print(f"[WARN] 接受率仍低，建议检查：")
        print(f"  1. mask质量——鸭子点云太少（<200点）时FPFH特征不可靠")
        print(f"  2. 适当增大 --kf_interval（当前{keyframe_interval}→试试5）")
        print(f"  3. 调小 icp_voxel（当前{icp_voxel:.4f}），增加特征点密度")
 
    pts = np.asarray(merged.points, dtype=np.float32)
    if merged.has_colors():
        cols = (np.asarray(merged.colors) * 255).astype(np.float32)
        return np.hstack([pts, cols])
    return pts


# ══════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    # 1. 基础参数
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="duck_fixed.pcd")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--no_ransac", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--no_global_reg", action="store_true")

    # 2. 核心配准参数 (刚才报错就是因为缺这几行)
    parser.add_argument("--max_trans", type=float, default=0.05, help="每帧最大平移")
    parser.add_argument("--max_angle", type=float, default=8.0, help="每帧最大旋转")
    parser.add_argument("--kf_interval", type=int, default=8, help="关键帧间隔")
    parser.add_argument("--dbscan_eps", type=float, default=0.05, help="DBSCAN半径")

    args = parser.parse_args()

    # --- 后面的逻辑开始 ---
    print("=" * 60)
    print("  Duck PCD Pipeline - v3 (Duck-to-Duck ICP)")
    print("=" * 60)

    # 读数据
    reader = BagReader(args.bag)
    frames = reader.read_all(max_frames=args.max_frames)
    if not frames:
        print("[ERROR] 没有配对到帧")
        sys.exit(1)
    if args.skip_frames > 1:
        frames = frames[::args.skip_frames]
        print(f"[INFO] 跳帧后 {len(frames)} 帧")

    frames_rgb = [f[0] for f in frames]
    frames_pcd = [f[1] for f in frames]
    print(f"[INFO] RGB: {frames_rgb[0].shape}, 首帧点数: {len(frames_pcd[0])}")

    # 读相机参数
    color_intr = get_color_intrinsics(args.bag)
    extrinsics = get_extrinsics_depth_to_color(args.bag)
    print(f"[TF] 外参矩阵:\n{np.round(extrinsics, 4)}")

    # SAM2分割
    seg = DuckSegmenter(args.sam2_checkpoint)
    pos_pts, neg_pts = seg.get_clicks(frames_rgb[0])
    print(f"[SEG] 正样本: {len(pos_pts)}, 负样本: {len(neg_pts)}")
    masks = seg.segment_all(frames_rgb, pos_pts, neg_pts)

    # 保存mask预览
    if args.save_masks:
        os.makedirs("masks_preview", exist_ok=True)
        for i in range(min(200, len(masks))):
            prev = frames_rgb[i].copy()
            prev[masks[i]] = (prev[masks[i]] * 0.4 +
                              np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            cv2.imwrite(f"masks_preview/frame_{i:04d}.jpg",
                       cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
        print("[INFO] mask预览已保存到 masks_preview/")

    # 逐帧提取鸭子点云
    valid_ducks = []
    print(f"\n[PCD] 逐帧过滤，共 {len(frames)} 帧...")
    for i, (pcd_raw, mask) in enumerate(zip(frames_pcd, masks)):
        if mask is None or mask.sum() < 100:
            continue
        h, w = frames_rgb[i].shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)

        duck = filter_duck_points(
            pcd_raw, frames_rgb[i], mask, color_intr, extrinsics)
        if len(duck) < 20:
            continue

        duck = voxel_down(duck, args.voxel_size)
        valid_ducks.append(duck)

        if i % 50 == 0:
            print(f"[PCD] 帧{i}: 鸭子{len(duck)}点, mask{mask.sum()}px")

    print(f"\n[PCD] 有效帧: {len(valid_ducks)}")
    if not valid_ducks:
        print("[ERROR] 没有提取到鸭子点云！")
        sys.exit(1)

    # ICP多帧配准融合（新版 Duck-to-Duck）
    if len(valid_ducks) == 1:
        merged = valid_ducks[0]
        print("[INFO] 只有一帧，跳过ICP")
    else:
        merged = merge_with_global_target(
            valid_ducks,
            voxel_size_duck=args.voxel_size,
            max_z_shift=0.05,
            min_fitness=0.40,
            batch_size=5)

    if len(merged) == 0:
        print("[ERROR] ICP后点云为空")
        sys.exit(1)

    print(f"[PCD] ICP后点数: {len(merged)}")

    # 统计滤波（去孤立噪点）
    merged = statistical_filter(merged, nb=20, std=2.0)

    # RANSAC去桌面
    if not args.no_ransac:
        merged = remove_floor_ransac(merged, dist_thresh=0.012)

    # ★ DBSCAN聚类：强制保留最大连通块（关键步骤）
    merged = keep_largest_cluster(merged, eps=args.dbscan_eps, min_pts=10)

    # 再次统计滤波
    merged = statistical_filter(merged, nb=15, std=1.5)

    # 最终降采样
    merged = voxel_down(merged, args.voxel_size)
    print(f"[PCD] 最终点数: {len(merged)}")

    # 保存
    final = o3d.geometry.PointCloud()
    final.points = o3d.utility.Vector3dVector(merged[:, :3].astype(np.float64))
    if merged.shape[1] >= 6:
        final.colors = o3d.utility.Vector3dVector(
            np.clip(merged[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    final.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    final.orient_normals_consistent_tangent_plane(15)
    o3d.io.write_point_cloud(args.output, final)

    pts = np.asarray(final.points)
    size_x = pts[:, 0].max() - pts[:, 0].min()
    size_y = pts[:, 1].max() - pts[:, 1].min()
    size_z = pts[:, 2].max() - pts[:, 2].min()
    print(f"\n✅ 保存到: {args.output}")
    print(f"   总点数: {len(pts)}")
    print(f"   X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m  (跨度 {size_x:.3f}m)")
    print(f"   Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m  (跨度 {size_y:.3f}m)")
    print(f"   Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m  (跨度 {size_z:.3f}m)")
    if max(size_x, size_y, size_z) > 0.5:
        print("⚠️  点云跨度 > 50cm，可能仍有漂移帧混入。")
        print("   建议：降低 --max_trans（当前{:.2f}）或 --max_angle（当前{:.1f}）".format(
            args.max_trans, args.max_angle))


if __name__ == "__main__":
    main()
  