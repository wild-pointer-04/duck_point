import os
import argparse
import numpy as np
import open3d as o3d
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_typesfrommsg

# 定义 sensor_msgs/msg/PointCloud2 的结构，用于解析原始二进制数据
def get_pcd_dtype(fields):
    dtype = []
    for field in fields:
        if field.datatype == 7: # FLOAT32
            dtype.append((field.name, np.float32))
        elif field.datatype == 4: # UINT16 (intensity etc.)
            dtype.append((field.name, np.uint16))
    return np.dtype(dtype)

def main():
    parser = argparse.ArgumentParser(description="Extract PointCloud2 from ROS2 bag to PCD.")
    parser.add_argument("bag_path", help="Path to the .db3 file")
    parser.add_argument("output_dir", help="Directory to save .pcd files")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    topic_name = '/camera/depth/points'
    count = 0

    print(f"Reading bag from {args.bag_path}...")
    with Reader(args.bag_path) as reader:
        # 预先检查话题是否存在
        conns = [c for c in reader.connections if c.topic == topic_name]
        if not conns:
            print(f"Topic {topic_name} not found in bag!")
            return
        
        # 遍历消息
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            # 反序列化
            msg = deserialize_cdr(rawdata, connection.msgtype)
            
            # 动态获取数据类型（处理不同相机的字段差异）
            dtype = get_pcd_dtype(msg.fields)
            
            # 解析二进制数据
            pc_data = np.frombuffer(msg.data, dtype=dtype)
            
            # 提取 X, Y, Z (处理可能存在的 NaN 值)
            points = np.zeros((pc_data.shape[0], 3), dtype=np.float32)
            points[:, 0] = pc_data['x']
            points[:, 1] = pc_data['y']
            points[:, 2] = pc_data['z']
            
            # 移除 NaN 值（很多 ToF/深度相机会有 NaN）
            mask = ~np.isnan(points).any(axis=1)
            final_points = points[mask]

            if final_points.size == 0:
                continue

            # 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(final_points)
            
            # 如果原始数据有颜色(rgb字段)，也可以在这里提取并赋值
            # if 'rgb' in pc_data.dtype.names:
            #    # 这里需要复杂的位域解析过程，省略

            # 保存为 PCD
            filename = os.path.join(args.output_dir, f"duck_frame_{timestamp}.pcd")
            o3d.io.write_point_cloud(filename, pcd)
            
            count += 1
            if count % 10 == 0:
                print(f"Extracted {count} frames...", end='\r')

    print(f"\nSuccessfully extracted {count} PCD files to '{args.output_dir}'.")

if __name__ == "__main__":
    main()