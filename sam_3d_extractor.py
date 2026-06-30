import os
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- 替换为 ROS 2 原生导入 ---
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# 全局变量
ix, iy = -1, -1
drawing = False

def draw_point(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print(f"点击坐标: ({ix}, {iy})")

def depth_to_pcd(depth_image, fx, fy, cx, cy):
    h, w = depth_image.shape
    u = np.arange(0, w)
    v = np.arange(0, h)
    uu, vv = np.meshgrid(u, v)
    z = depth_image.astype(np.float32) / 1000.0 # 假设单位是mm
    mask = (z > 0.1) & (z < 2.0)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    points = np.stack([x[mask], y[mask], z[mask]], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("sam_checkpoint")
    args = parser.parse_args()

    # 1. 加载 SAM
    print("正在加载 SAM 模型...")
    device = "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 2. 使用 ROS 2 原生 Reader 读取 Bag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: get_message(t.type) for t in topic_types}

    color_topic = '/camera/color/image_raw'
    depth_topic = '/camera/depth/image_raw'
    
    rgb_img, depth_img = None, None

    print("正在搜索图像帧...")
    while reader.has_next() and (rgb_img is None or depth_img is None):
        topic, data, t = reader.read_next()
        if topic == color_topic and rgb_img is None:
            msg = deserialize_message(data, type_map[color_topic])
            rgb_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        if topic == depth_topic and depth_img is None:
            msg = deserialize_message(data, type_map[depth_topic])
            depth_img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    if rgb_img is None: 
        print("未找到图像"); return

    # 3. 交互点击
    cv2.imshow('Click Duck, then SPACE', rgb_img)
    cv2.setMouseCallback('Click Duck, then SPACE', draw_point)
    print("请在图像窗口中点击鸭子中心...")
    print("点击后，按下键盘上的 空格键(SPACE) 确认。")

    # 使用循环来保持窗口响应，直到按下空格键
    while True:
        cv2.imshow('Click Duck, then SPACE', rgb_img) # 持续刷新显示（如果有需要）
        key = cv2.waitKey(1) & 0xFF # 关键：用1ms等待代替0，保持响应
        
        # 如果 ix 已经被点击了（不再是-1），并且按下了空格键
        if ix != -1 and key == ord(' '):
            print(f"确认点击坐标: ({ix}, {iy})")
            break
        elif key == 27: # 按 ESC 键可以强制退出
            print("用户取消操作。")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # 4. SAM 分割
    predictor.set_image(rgb_img)
    masks, scores, _ = predictor.predict(np.array([[ix, iy]]), np.array([1]), multimask_output=True)
    mask = masks[np.argmax(scores)]

    # --- 5. 修复分辨率不匹配问题 ---
    print(f"深度图尺寸: {depth_img.shape}, Mask尺寸: {mask.shape}")
    
    # 将布尔类型的 Mask 转换为 uint8，然后缩放
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_uint8, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 转回布尔类型
    mask_final = mask_resized > 0

    # 使用缩放后的 Mask 过滤深度图
    pcd = depth_to_pcd(depth_img * mask_final, 570.34, 570.34, 320.0, 240.0)

    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("duck_final.pcd", pcd)

if __name__ == "__main__":
    main()