#!/usr/bin/env python3
"""
提取第一帧目标物体点云 (结合 SAM2 分割与 TF 投影)
"""

import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d
from collections import deque
import torch
from sam2.build_sam import build_sam2_video_predictor

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# ══════════════════════════════════════════
# 1. 工具函数与 Bag 解析 (复用你的逻辑)
# ══════════════════════════════════════════
def _extract_field_fast(data, offset, step, n, dtype):
    itemsize = np.dtype(dtype).itemsize
    row_starts = np.arange(n) * step + offset
    indices = (row_starts[:, None] + np.arange(itemsize)[None, :]).ravel()
    return np.frombuffer(data[indices].tobytes(), dtype=dtype)

class BagReader:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        sr = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        cr = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(sr, cr)
        self.type_map = {t.name: t.type for t in self.reader.get_all_topics_and_types()}

    def read_first_frame(self):
        """只读取并配对第一帧数据"""
        color_buf, points_buf = deque(), deque()
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

            # 尝试配对
            while color_buf and points_buf:
                tc, ic = color_buf[0]
                tp, pp = points_buf[0]
                dt = abs(tc - tp) / 1e6
                if dt < 200.0: # 时间差在 200ms 以内认为同步
                    return ic, pp, tc # 成功配对第一帧直接返回
                elif tc < tp:
                    color_buf.popleft()
                else:
                    points_buf.popleft()
        return None, None, None

    @staticmethod
    def _decode_image(msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding.lower() in ("rgb8", "rgb"):
            return data.reshape(msg.height, msg.width, 3)
        elif msg.encoding.lower() in ("bgr8", "bgr"):
            return data.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
        return None

    @staticmethod
    def _decode_pointcloud2(msg):
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")): return None
        step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        n = msg.width * msg.height
        result = np.zeros((n, 6), dtype=np.float32)
        for i, name in enumerate(["x", "y", "z"]):
            result[:, i] = _extract_field_fast(data, fields[name].offset, step, n, np.float32)
        if "rgb" in fields:
            raw = _extract_field_fast(data, fields["rgb"].offset, step, n, np.float32)
            rgb_int = raw.view(np.uint32)
            result[:, 3] = ((rgb_int >> 16) & 0xFF).astype(np.float32)
            result[:, 4] = ((rgb_int >> 8) & 0xFF).astype(np.float32)
            result[:, 5] = (rgb_int & 0xFF).astype(np.float32)
        else:
            result[:, 3:] = 200.0
        # 过滤无效点
        valid = np.isfinite(result[:, :3]).all(axis=1) & (result[:, 2] > 0.01)
        return result[valid]

# --- 获取相机参数逻辑与你的完全一致，此处为了紧凑直接放过来 ---
def get_color_intrinsics(bag_path):
    # [简化] 这里为了代码紧凑，直接提取你的默认值，如果是真实使用，可以用你原来的完整函数提取
    return (691.33, 691.51, 643.92, 362.12, 1280, 720)

def get_extrinsics_depth_to_color(bag_path):
    # [简化] 这里填入你在日志中打印出来的TF变换，平移=[-0.1, -2.0, 13.4]mm
    mat = np.eye(4, dtype=np.float64)
    mat[0, 3] = -0.0001
    mat[1, 3] = -0.0020
    mat[2, 3] = 0.0134
    return mat

# ══════════════════════════════════════════
# 2. SAM2 & 点云过滤 (复用你的逻辑)
# ══════════════════════════════════════════
class DuckSegmenter:
    def __init__(self, sam2_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = "configs/sam2.1/sam2.1_hiera_t.yaml" if "tiny" in sam2_checkpoint else "configs/sam2.1/sam2.1_hiera_s.yaml"
        self.predictor = build_sam2_video_predictor(cfg, sam2_checkpoint, device=device)

    def get_clicks_and_segment(self, rgb_image):
        h, w = rgb_image.shape[:2]
        disp = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        scale = min(1.0, 1280/w, 720/h)
        disp_resized = cv2.resize(disp, (int(w*scale), int(h*scale)))
        pos, neg = [], []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([x/scale, y/scale])
            elif event == cv2.EVENT_RBUTTONDOWN:
                neg.append([x/scale, y/scale])

        cv2.namedWindow("Click & Segment", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Click & Segment", on_mouse)
        
        while True:
            img = disp_resized.copy()
            for p in pos: cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,255,0), -1)
            for p in neg: cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)), 8, (0,0,255), -1)
            cv2.putText(img, "Left=Pos, Right=Neg, ENTER=Confirm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Click & Segment", img)
            if cv2.waitKey(30) & 0xFF == 13 and pos: # Enter
                break
        cv2.destroyAllWindows()

        # 对单帧进行推理 (利用视频推理器的结构)
        import tempfile, shutil
        from PIL import Image
        tmp = tempfile.mkdtemp(prefix="sam2_")
        Image.fromarray(rgb_image).save(os.path.join(tmp, "00000.jpg"))
        with torch.inference_mode():
            state = self.predictor.init_state(video_path=tmp)
            pts = np.vstack([pos, neg]).astype(np.float32)
            lbs = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
            self.predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, points=pts, labels=lbs)
            mask = None
            for idx, _, logits in self.predictor.propagate_in_video(state):
                mask = (logits[0, 0] > 0.0).cpu().numpy()
        shutil.rmtree(tmp, ignore_errors=True)
        return mask

def filter_points_by_mask(pcd_nx6, rgb_image, mask, color_intrinsics, extrinsics_d2c):
    fx, fy, cx, cy, W, H = color_intrinsics
    pts3d = pcd_nx6[:, :3].astype(np.float64)
    N = len(pts3d)

    pts_h = np.hstack([pts3d, np.ones((N, 1))])
    pts_c = (extrinsics_d2c @ pts_h.T).T
    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]

    valid_z = Zc > 0.05
    u = np.full(N, -1, dtype=np.int32)
    v = np.full(N, -1, dtype=np.int32)
    u[valid_z] = np.round(Xc[valid_z] / Zc[valid_z] * fx + cx).astype(np.int32)
    v[valid_z] = np.round(Yc[valid_z] / Zc[valid_z] * fy + cy).astype(np.int32)

    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    in_mask = np.zeros(N, dtype=bool)
    idx = np.where(in_bounds)[0]
    in_mask[idx] = mask[v[idx], u[idx]]

    result = pcd_nx6[in_mask].copy()
    if result.shape[1] >= 6 and len(result) > 0:
        idx_kept = np.where(in_mask)[0]
        result[:, 3] = rgb_image[v[idx_kept], u[idx_kept], 0].astype(np.float32)
        result[:, 4] = rgb_image[v[idx_kept], u[idx_kept], 1].astype(np.float32)
        result[:, 5] = rgb_image[v[idx_kept], u[idx_kept], 2].astype(np.float32)
    return result

# ══════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--output", default="first_frame_bottle.pcd")
    args = parser.parse_args()

    print("[1] 正在读取 Bag 的第一帧...")
    reader = BagReader(args.bag)
    rgb_img, pcd_raw, _ = reader.read_first_frame()
    
    if rgb_img is None or pcd_raw is None:
        print("[ERROR] 无法读取或配对图像和点云。")
        return
    print(f"成功获取第一帧! 图像大小: {rgb_img.shape}, 原始点云数量: {len(pcd_raw)}")

    color_intr = get_color_intrinsics(args.bag)
    extrinsics = get_extrinsics_depth_to_color(args.bag)

    print("[2] 启动 SAM2 交互...")
    seg = DuckSegmenter(args.sam2_checkpoint)
    mask = seg.get_clicks_and_segment(rgb_img)
    
    if mask is None or mask.sum() < 50:
        print("[ERROR] Mask 提取失败或目标过小。")
        return

    print(f"[3] 正在根据 Mask 提取 3D 点云... (Mask像素数: {mask.sum()})")
    bottle_pts = filter_points_by_mask(pcd_raw, rgb_img, mask, color_intr, extrinsics)
    print(f"提取出 {len(bottle_pts)} 个目标点。")

    if len(bottle_pts) == 0:
        print("[WARN] 提取出的点云为空！可能是相机盲区或外参不对齐。")
        return

    # 简单统计滤波去噪
    print("[4] 执行统计滤波并保存...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bottle_pts[:, :3].astype(np.float64))
    if bottle_pts.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(bottle_pts[:, 3:6] / 255.0, 0, 1).astype(np.float64))
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"\n✅ 提取完毕！最终水瓶点数: {len(pcd.points)}")
    print(f"文件保存在: {args.output}")

if __name__ == "__main__":
    main()