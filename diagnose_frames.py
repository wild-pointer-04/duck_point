#!/usr/bin/env python3
"""
诊断各帧点云的实际密度、深度范围、以及 mask 对应区域的点云覆盖率
"""
import sys
import numpy as np
import cv2
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


def decode_image(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "rgb"):
        return data.reshape(msg.height, msg.width, 3)
    elif enc in ("bgr8", "bgr"):
        return data.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    return None


def decode_pcd(msg):
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return None
    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    n = msg.width * msg.height
    xyz = np.zeros((n, 3), dtype=np.float32)
    for i, name in enumerate(["x", "y", "z"]):
        xyz[:, i] = _extract_field_fast(data, fields[name].offset, step, n, np.float32)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.01)
    return xyz[valid]


def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_frames.py <bag_path>")
        sys.exit(1)

    bag_path = sys.argv[1]
    reader, type_map = open_bag(bag_path)

    # 相机内参（已知）
    fx, fy, cx, cy, W, H = 691.33, 691.51, 643.92, 362.12, 1280, 720

    color_buf, depth_buf = deque(), deque()
    frame_idx = 0
    CHECK_FRAMES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]  # 重点检查这些帧

    paired = []

    while reader.has_next():
        topic, raw, ts = reader.read_next()
        if topic == "/camera/color/image_raw":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            img = decode_image(msg)
            if img is not None:
                color_buf.append((ts, img))
        elif topic == "/camera/depth/points":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            pts = decode_pcd(msg)
            if pts is not None:
                depth_buf.append((ts, pts))

        while color_buf and depth_buf:
            tc, ic = color_buf[0]
            tp, pp = depth_buf[0]
            dt = abs(tc - tp) / 1e6
            if dt < 300:
                paired.append((ic, pp))
                color_buf.popleft()
                depth_buf.popleft()
            elif tc < tp:
                color_buf.popleft()
            else:
                depth_buf.popleft()

    print(f"总配对帧数: {len(paired)}\n")
    print(f"{'帧':>4}  {'总点数':>8}  {'Z_min':>7}  {'Z_max':>7}  {'Z_mean':>7}  {'Z<0.5m点数':>10}  {'Z<0.3m点数':>10}")
    print("-" * 75)

    for i, (rgb, pcd) in enumerate(paired):
        if i not in CHECK_FRAMES:
            continue

        Z = pcd[:, 2]
        n_total = len(pcd)
        z_min = Z.min()
        z_max = Z.max()
        z_mean = Z.mean()
        n_close = (Z < 0.5).sum()
        n_very_close = (Z < 0.3).sum()

        print(f"{i:>4}  {n_total:>8}  {z_min:>7.3f}  {z_max:>7.3f}  {z_mean:>7.3f}  {n_close:>10}  {n_very_close:>10}")

        # 对关键帧（70, 80, 90）可视化深度图
        if i in [0, 70, 80, 90]:
            # 把点云投影回图像，画出深度热力图
            X, Y, Z2 = pcd[:, 0], pcd[:, 1], pcd[:, 2]
            valid = Z2 > 0.05
            u = np.round(X[valid] / Z2[valid] * fx + cx).astype(np.int32)
            v = np.round(Y[valid] / Z2[valid] * fy + cy).astype(np.int32)
            ib = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z_vis = u[ib], v[ib], Z2[valid][ib]

            depth_img = np.zeros((H, W), dtype=np.float32)
            depth_img[v, u] = z_vis
            # 归一化为热力图
            dmax = np.percentile(z_vis, 95) if len(z_vis) > 0 else 1.0
            dmin = z_vis.min() if len(z_vis) > 0 else 0.0
            norm = np.clip((depth_img - dmin) / (dmax - dmin + 1e-6), 0, 1)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # mask 掉深度为 0 的区域
            heat[depth_img == 0] = 0

            # 并排显示彩色图 + 深度热力图
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            scale = 0.5
            rgb_s = cv2.resize(rgb_bgr, (int(W*scale), int(H*scale)))
            heat_s = cv2.resize(heat, (int(W*scale), int(H*scale)))
            combined = np.hstack([rgb_s, heat_s])
            cv2.putText(combined, f"Frame {i} | Points:{n_total} | Z:{z_min:.2f}~{z_max:.2f}m",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(f"Frame {i}: RGB | Depth Heatmap (close=red far=blue)", combined)
            print(f"  → 按任意键继续...")
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # 额外检查：相邻帧点云数量变化趋势
    print("\n[全帧点云数量趋势]")
    print(f"{'帧':>4}  {'点数':>8}  {'Z_mean':>7}")
    for i, (rgb, pcd) in enumerate(paired):
        print(f"{i:>4}  {len(pcd):>8}  {pcd[:,2].mean():>7.3f}")


if __name__ == "__main__":
    main()