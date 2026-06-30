#!/usr/bin/env python3
"""
诊断脚本：
1. 检查 /camera/depth/points 的 frame_id
2. 读取 /tf_static 获取真实外参
3. 可视化投影对齐效果（把点云投影到彩色图上）
"""
 
import numpy as np
import cv2
import sys
from collections import deque
 
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
 
 
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)
 
 
def open_bag(bag_path):
    sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    cr = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(sr, cr)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, type_map
 
 
def read_first_frame_with_meta(bag_path):
    """读取第一对配对帧，同时打印关键 meta 信息"""
    reader, type_map = open_bag(bag_path)
 
    color_buf, depth_buf = deque(), deque()
    pc_frame_id = None
    color_frame_id = None
    result = {"rgb": None, "pcd": None, "K": None, "pc_frame_id": None, "color_frame_id": None}
 
    while reader.has_next():
        topic, raw, ts = reader.read_next()
 
        if topic == "/camera/color/camera_info" and result["K"] is None:
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = np.array(msg.k).reshape(3, 3)
            result["K"] = K
            result["W"] = msg.width
            result["H"] = msg.height
            result["color_frame_id"] = msg.header.frame_id
            print(f"\n[CameraInfo] frame_id = '{msg.header.frame_id}'")
            print(f"[CameraInfo] K =\n{K}")
            print(f"[CameraInfo] 图像尺寸: {msg.width}x{msg.height}")
 
        elif topic == "/camera/depth/points":
            msg = deserialize_message(raw, get_message(type_map[topic]))
 
            if pc_frame_id is None:
                pc_frame_id = msg.header.frame_id
                result["pc_frame_id"] = pc_frame_id
                print(f"\n[PointCloud2] frame_id = '{pc_frame_id}'")
                print(f"[PointCloud2] fields = {[f.name for f in msg.fields]}")
                print(f"[PointCloud2] point_step = {msg.point_step}, width={msg.width}, height={msg.height}")
 
            # 解析点云
            fields = {f.name: f for f in msg.fields}
            step = msg.point_step
            data = np.frombuffer(msg.data, dtype=np.uint8)
            n = msg.width * msg.height
            xyz = np.zeros((n, 3), dtype=np.float32)
            for i, name in enumerate(["x", "y", "z"]):
                xyz[:, i] = _extract_field_fast(data, fields[name].offset, step, n, np.float32)
            valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.01)
            xyz = xyz[valid]
            depth_buf.append((ts, xyz))
 
        elif topic == "/camera/color/image_raw":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            data = np.frombuffer(msg.data, dtype=np.uint8)
            enc = msg.encoding.lower()
            if enc in ("rgb8", "rgb"):
                img = data.reshape(msg.height, msg.width, 3)
            elif enc in ("bgr8", "bgr"):
                img = data.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
            else:
                continue
            color_buf.append((ts, img))
 
        # 配对
        while color_buf and depth_buf:
            tc, ic = color_buf[0]
            tp, pp = depth_buf[0]
            dt = abs(tc - tp) / 1e6
            if dt < 300:
                result["rgb"] = ic
                result["pcd"] = pp
                print(f"\n[配对成功] 时间差 {dt:.1f}ms，点数 {len(pp)}")
                return result
            elif tc < tp:
                color_buf.popleft()
            else:
                depth_buf.popleft()
 
    return result
 
 
def read_tf_static(bag_path):
    """读取 /tf_static，打印所有变换"""
    reader, type_map = open_bag(bag_path)
    print("\n" + "="*60)
    print("[TF Static] 所有静态变换：")
    found = False
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/tf_static":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            for tf in msg.transforms:
                t = tf.transform.translation
                r = tf.transform.rotation
                print(f"  {tf.header.frame_id} -> {tf.child_frame_id}")
                print(f"    平移: x={t.x:.6f} y={t.y:.6f} z={t.z:.6f}")
                print(f"    四元数: x={r.x:.6f} y={r.y:.6f} z={r.z:.6f} w={r.w:.6f}")
                found = True
    if not found:
        print("  [未找到任何 tf_static！]")
    return found
 
 
def quat_to_rotation_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵"""
    R = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ])
    return R
 
 
def visualize_projection(rgb, pcd_xyz, K, W, H, extrinsic_4x4, title="投影可视化"):
    """将点云投影到彩色图上，检查对齐效果"""
    N = len(pcd_xyz)
    pts_h = np.hstack([pcd_xyz, np.ones((N, 1), dtype=np.float64)])
    pts_c = (extrinsic_4x4 @ pts_h.T).T
    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
 
    valid = Zc > 0.05
    u = np.full(N, -1, dtype=np.int32)
    v = np.full(N, -1, dtype=np.int32)
    u[valid] = np.round(Xc[valid] / Zc[valid] * K[0,0] + K[0,2]).astype(np.int32)
    v[valid] = np.round(Yc[valid] / Zc[valid] * K[1,1] + K[1,2]).astype(np.int32)
 
    in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print(f"\n[投影] 总点数={N}, 有效Z={valid.sum()}, 落在图像内={in_bounds.sum()}")
 
    # 深度图（用于判断点分布范围）
    depths = Zc[in_bounds]
    u_vis = u[in_bounds]
    v_vis = v[in_bounds]
 
    if len(depths) > 0:
        print(f"[投影] 深度范围: {depths.min():.3f}m ~ {depths.max():.3f}m")
        print(f"[投影] u范围: {u_vis.min()} ~ {u_vis.max()}")
        print(f"[投影] v范围: {v_vis.min()} ~ {v_vis.max()}")
 
    # 可视化
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    # 按深度着色
    if len(depths) > 0:
        d_min, d_max = depths.min(), depths.max()
        for uu, vv, dd in zip(u_vis[::5], v_vis[::5], depths[::5]):  # 每5个画一个点
            norm = (dd - d_min) / (d_max - d_min + 1e-6)
            color = (int(norm * 255), int((1-norm) * 255), 0)
            cv2.circle(vis, (uu, vv), 2, color, -1)
 
    scale = min(1.0, 1280/W, 720/H)
    vis_small = cv2.resize(vis, (int(W*scale), int(H*scale)))
    cv2.imshow(title, vis_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return in_bounds.sum()
 
 
def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_projection.py <bag_path>")
        sys.exit(1)
 
    bag_path = sys.argv[1]
 
    print("="*60)
    print(f"诊断 bag: {bag_path}")
    print("="*60)
 
    # 1. 读取TF静态变换
    read_tf_static(bag_path)
 
    # 2. 读取第一帧数据
    print("\n" + "="*60)
    print("[读取第一帧]")
    result = read_first_frame_with_meta(bag_path)
 
    if result["rgb"] is None or result["pcd"] is None:
        print("[ERROR] 未能读取到配对帧")
        return
 
    rgb = result["rgb"]
    pcd = result["pcd"].astype(np.float64)
    K = result["K"]
    W, H = result["W"], result["H"]
 
    print(f"\n[点云坐标统计]")
    print(f"  X: {pcd[:,0].min():.3f} ~ {pcd[:,0].max():.3f} m")
    print(f"  Y: {pcd[:,1].min():.3f} ~ {pcd[:,1].max():.3f} m")
    print(f"  Z: {pcd[:,2].min():.3f} ~ {pcd[:,2].max():.3f} m")
 
    # 3. 尝试不同外参方案
    print("\n" + "="*60)
    print("[方案A] 外参=单位阵（假设点云已经在color坐标系）")
    eye = np.eye(4, dtype=np.float64)
    n_a = visualize_projection(rgb, pcd, K, W, H, eye, "方案A: 单位阵外参")
 
    print("\n[方案B] 外参=你原来的微小平移")
    T_b = np.eye(4, dtype=np.float64)
    T_b[0,3] = -0.0001; T_b[1,3] = -0.0020; T_b[2,3] = 0.0134
    n_b = visualize_projection(rgb, pcd, K, W, H, T_b, "方案B: 微小平移外参")
 
    print(f"\n[结论] 方案A落点数={n_a}, 方案B落点数={n_b}")
    print("→ 落点数更多且图像对齐更好的方案是正确的")
    print("→ 如果点云frame_id已经是color_optical_frame，用方案A（单位阵）")
 
 
if __name__ == "__main__":
    main()