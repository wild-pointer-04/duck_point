#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
duck_sam2_rosbag_TSDF_VO_FINAL.py
======================================================
ROS2 Bag + SAM2 + RGBD Visual Odometry + TSDF Final

解决上一版碎裂问题：
❌ 假设直线推进 forward_step
✅ 改为 Open3D RGBD Visual Odometry 自动估计相机位姿

适合：
- 手持相机靠近物体
- 有轻微抖动/偏移
- 水瓶/杯子/鸭子/盒子

输出：
output_vo/
    object_mesh.ply
    object_dense.pcd
    preview_masks/

======================================================
"""

import os
import cv2
import argparse
import numpy as np
import open3d as o3d
from collections import deque

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

import torch
from sam2.build_sam import build_sam2_video_predictor


# ======================================================
# Utils
# ======================================================
def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


# ======================================================
# ROSBAG Reader
# ======================================================
class BagReader:
    def __init__(self, bag):
        self.bag = bag

        storage = rosbag2_py.StorageOptions(
            uri=bag,
            storage_id="sqlite3"
        )

        conv = rosbag2_py.ConverterOptions("", "")

        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage, conv)

        self.type_map = {
            t.name: t.type
            for t in self.reader.get_all_topics_and_types()
        }

    def decode_rgb(self, msg):
        h, w = msg.height, msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()

        if enc in ["rgb8", "rgb"]:
            return data.reshape(h, w, 3)

        if enc in ["bgr8", "bgr"]:
            return data.reshape(h, w, 3)[:, :, ::-1].copy()

        return None

    def decode_depth(self, msg):
        h, w = msg.height, msg.width
        enc = msg.encoding.lower()

        if enc == "16uc1":
            depth = np.frombuffer(
                msg.data, dtype=np.uint16
            ).reshape(h, w).astype(np.float32)

            return depth / 1000.0

        if enc == "32fc1":
            return np.frombuffer(
                msg.data, dtype=np.float32
            ).reshape(h, w)

        return None

    def read_pairs(self, max_frames=None):
        rgb_buf = deque()
        dep_buf = deque()

        pairs = []

        while self.reader.has_next():
            topic, raw, ts = self.reader.read_next()

            if topic == "/camera/color/image_raw":
                msg = deserialize_message(raw, get_message(self.type_map[topic]))
                img = self.decode_rgb(msg)
                if img is not None:
                    rgb_buf.append((ts, img))

            elif topic == "/camera/depth/image_raw":
                msg = deserialize_message(raw, get_message(self.type_map[topic]))
                dep = self.decode_depth(msg)
                if dep is not None:
                    dep_buf.append((ts, dep))

            while rgb_buf and dep_buf:
                tr, rgb = rgb_buf[0]
                td, dep = dep_buf[0]

                dt = abs(tr - td) / 1e6

                if dt < 80:
                    pairs.append((rgb, dep))
                    rgb_buf.popleft()
                    dep_buf.popleft()

                    if max_frames and len(pairs) >= max_frames:
                        return pairs

                elif tr < td:
                    rgb_buf.popleft()
                else:
                    dep_buf.popleft()

        return pairs


# ======================================================
# Intrinsics
# ======================================================
def get_intrinsics(bag):
    storage = rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, rosbag2_py.ConverterOptions("", ""))

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    while reader.has_next():
        topic, raw, _ = reader.read_next()

        if topic == "/camera/color/camera_info":
            msg = deserialize_message(raw, get_message(type_map[topic]))
            K = msg.k

            return (
                K[0], K[4], K[2], K[5],
                msg.width, msg.height
            )

    raise RuntimeError("camera_info not found")


# ======================================================
# SAM2
# ======================================================
class Segmenter:
    def __init__(self, ckpt):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "tiny" in ckpt:
            cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "small" in ckpt:
            cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "base_plus" in ckpt:
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.predictor = build_sam2_video_predictor(
            cfg, ckpt, device=device
        )

    def clicks(self, rgb):
        h, w = rgb.shape[:2]
        scale = min(1.0, 1280 / w, 720 / h)

        disp = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        pos = []
        neg = []

        def redraw():
            img = disp.copy()

            for p in pos:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)),
                           7, (0,255,0), -1)

            for p in neg:
                cv2.circle(img, (int(p[0]*scale), int(p[1]*scale)),
                           7, (0,0,255), -1)

            cv2.putText(
                img,
                "LEFT=POS RIGHT=NEG ENTER=OK",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2
            )
            return img

        def mouse(event, x, y, flags, param):
            rx = x / scale
            ry = y / scale

            if event == cv2.EVENT_LBUTTONDOWN:
                pos.append([rx, ry])

            if event == cv2.EVENT_RBUTTONDOWN:
                neg.append([rx, ry])

        cv2.namedWindow("SAM2", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("SAM2", mouse)

        while True:
            cv2.imshow("SAM2", cv2.cvtColor(redraw(), cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(30) & 0xFF

            if k == 13 and len(pos) > 0:
                break

        cv2.destroyAllWindows()

        return np.array(pos), np.array(neg)

    def segment_video(self, rgbs, pos, neg):
        import tempfile
        import shutil
        from PIL import Image

        tmp = tempfile.mkdtemp(prefix="sam2_")

        try:
            for i, img in enumerate(rgbs):
                Image.fromarray(img).save(
                    os.path.join(tmp, f"{i:05d}.jpg")
                )

            state = self.predictor.init_state(video_path=tmp)

            pts = np.vstack([pos, neg]).astype(np.float32)
            lbs = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)

            self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=pts,
                labels=lbs
            )

            masks = [None] * len(rgbs)

            for idx, _, logits in self.predictor.propagate_in_video(state):
                masks[idx] = (logits[0,0] > 0).cpu().numpy()

            return masks

        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bag", required=True)
    parser.add_argument("--sam2_checkpoint", required=True)

    parser.add_argument("--max_frames", type=int, default=120)

    parser.add_argument("--voxel", type=float, default=0.0012)
    parser.add_argument("--min_depth", type=float, default=0.18)
    parser.add_argument("--max_depth", type=float, default=1.0)

    args = parser.parse_args()

    mkdir("output_vo")
    mkdir("output_vo/preview_masks")

    # -----------------------------------------
    # read data
    # -----------------------------------------
    reader = BagReader(args.bag)
    pairs = reader.read_pairs(args.max_frames)

    rgbs = [p[0] for p in pairs]
    deps = [p[1] for p in pairs]

    print("读取帧数:", len(rgbs))

    # -----------------------------------------
    # intrinsics
    # -----------------------------------------
    fx, fy, cx, cy, W, H = get_intrinsics(args.bag)

    intr = o3d.camera.PinholeCameraIntrinsic(
        W, H, fx, fy, cx, cy
    )

    # -----------------------------------------
    # SAM2
    # -----------------------------------------
    seg = Segmenter(args.sam2_checkpoint)

    pos, neg = seg.clicks(rgbs[0])
    masks = seg.segment_video(rgbs, pos, neg)

    # -----------------------------------------
    # TSDF
    # -----------------------------------------
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.voxel * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    odo_opt = o3d.pipelines.odometry.OdometryOption()

    global_pose = np.eye(4)

    prev_rgbd = None

    print("开始 VO + TSDF 融合...")

    for i in range(len(rgbs)):
        rgb = rgbs[i].copy()
        dep = deps[i].copy()
        mask = masks[i]

        if mask is None:
            continue

        dep[~mask] = 0
        dep[dep < args.min_depth] = 0
        dep[dep > args.max_depth] = 0

        if np.count_nonzero(dep) < 800:
            continue

        # preview
        if i < 30:
            prev = rgb.copy()
            prev[mask] = (
                prev[mask] * 0.4 +
                np.array([0,255,0]) * 0.6
            ).astype(np.uint8)

            cv2.imwrite(
                f"output_vo/preview_masks/{i:04d}.jpg",
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
            )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image(dep.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False
        )

        # 第一帧
        if prev_rgbd is None:
            volume.integrate(rgbd, intr, np.linalg.inv(global_pose))
            prev_rgbd = rgbd
            continue

        # -------------------------------------
        # RGBD Visual Odometry
        # -------------------------------------
        success, trans, info = \
            o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd,
                prev_rgbd,
                intr,
                np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                odo_opt
            )

        if success:
            global_pose = global_pose @ np.linalg.inv(trans)

        volume.integrate(rgbd, intr, np.linalg.inv(global_pose))

        prev_rgbd = rgbd

        if i % 10 == 0:
            print("融合帧", i, "VO:", success)

    # -----------------------------------------
    # extract
    # -----------------------------------------
    print("提取 mesh...")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh(
        "output_vo/object_mesh.ply", mesh
    )

    pcd = volume.extract_point_cloud()
    pcd = pcd.voxel_down_sample(args.voxel * 0.5)

    o3d.io.write_point_cloud(
        "output_vo/object_dense.pcd", pcd
    )

    print("完成，输出 output_vo/")


if __name__ == "__main__":
    main()