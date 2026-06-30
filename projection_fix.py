import cv2
import numpy as np
import argparse
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore  # 新版 API 的导入方式

def run_diagnostics(bag_path):
    print(f"正在分析 Bag: {bag_path}")

    rgb_msg = None
    pcd_msg = None
    info_msg = None

    # 初始化最新的类型系统 (适配新版 rosbags API)
    typestore = get_typestore(Stores.LATEST)

    # 1. 读取 Bag 文件，各取一帧
    try:
        with Reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/camera/color/image_raw' and rgb_msg is None:
                    # 使用 typestore 进行反序列化
                    rgb_msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                elif connection.topic == '/camera/depth/points' and pcd_msg is None:
                    pcd_msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                elif connection.topic == '/camera/color/camera_info' and info_msg is None:
                    info_msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                if rgb_msg and pcd_msg and info_msg:
                    break
    except Exception as e:
        print(f"读取 bag 失败: {e}")
        return

    if not all([rgb_msg, pcd_msg, info_msg]):
        print("错误：未能在 bag 中找到所需的所有话题！请检查话题名称。")
        return

    # 2. 诊断点云结构
    print("\n" + "="*30)
    print("--- 1. 点云结构诊断 ---")
    print(f"Height (高度): {pcd_msg.height}")
    print(f"Width (宽度): {pcd_msg.width}")
    print(f"Point Step (点步长): {pcd_msg.point_step}")
    print(f"Is Dense (是否稠密): {pcd_msg.is_dense}")

    if pcd_msg.height > 1:
        print("\n结论：这是【有序点云】(Structured Point Cloud)！可以通过 Mask 直接索引。")
    else:
        print("\n结论：这是【无序点云】(Unordered Point Cloud)！因为驱动剔除了无效点，必须使用内参重投影。")
    print("="*30)

    # 3. 解析相机内参
    K = info_msg.k
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    print(f"\n--- 2. RGB 相机内参 ---")
    print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 4. 提取 RGB 图像
    img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
    if rgb_msg.encoding in ['rgb8', '8UC3']:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 5. 提取 XYZ 点云数据
    print("\n--- 3. 开始执行重投影测试 ---")
    fields = {f.name: f.offset for f in pcd_msg.fields}
    xo, yo, zo = fields.get('x', 0), fields.get('y', 4), fields.get('z', 8)
    
    dt = np.dtype({
        'names': ['x', 'y', 'z'], 
        'formats': ['<f4', '<f4', '<f4'], 
        'offsets': [xo, yo, zo], 
        'itemsize': pcd_msg.point_step
    })
    
    cloud_arr = np.frombuffer(pcd_msg.data, dtype=dt)
    points = np.zeros((pcd_msg.width * pcd_msg.height, 3), dtype=np.float32)
    points[:, 0] = cloud_arr['x']
    points[:, 1] = cloud_arr['y']
    points[:, 2] = cloud_arr['z']

    # 6. 重投影到图像平面
    valid = points[:, 2] > 0  # 只保留 Z > 0 的点
    valid_points = points[valid]

    u = (valid_points[:, 0] * fx / valid_points[:, 2]) + cx
    v = (valid_points[:, 1] * fy / valid_points[:, 2]) + cy

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    h, w = img.shape[:2]
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid = u[mask]
    v_valid = v[mask]
    z_valid = valid_points[:, 2][mask]

    # 7. 绘制诊断图
    proj_img = np.zeros_like(img)
    z_norm = cv2.normalize(z_valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_z = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET).reshape(-1, 3)
    
    proj_img[v_valid, u_valid] = colored_z
    overlay = cv2.addWeighted(img, 0.4, proj_img, 0.6, 0)

    cv2.imwrite('diagnose_original_rgb.jpg', img)
    cv2.imwrite('diagnose_projection.jpg', overlay)
    print("\n✅ 测试完成！已生成诊断图像：")
    print("1. diagnose_original_rgb.jpg")
    print("2. diagnose_projection.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断点云结构与投影对齐")
    parser.add_argument("bag", help="你的 .db3 bag文件路径")
    args = parser.parse_args()
    
    run_diagnostics(args.bag)