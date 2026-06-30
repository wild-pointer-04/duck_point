import os
import argparse
import numpy as np
import open3d as o3d
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"--- 调试开始 ---")
    
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    
    reader.open(storage_options, converter_options)
    print("成功打开数据库！")

    topic_types = reader.get_all_topics_and_types()
    target_topic = '/camera/depth/points'
    msg_type_name = next((t.type for t in topic_types if t.name == target_topic), None)

    if not msg_type_name:
        print(f"错误：未找到 {target_topic}")
        return

    msg_type = get_message(msg_type_name)
    count = 0
    print("开始遍历消息...")

    # --- 关键修复：使用 read_next() 而不是 next() ---
    while reader.has_next():
        topic, data, t = reader.read_next() # Humble 版本必须用 read_next
        
        if topic == target_topic:
            msg = deserialize_message(data, msg_type)
            
            # 这里的解析逻辑针对奥比中光 PointCloud2
            # 步长通常是 32 字节 (x,y,z 为前 12 字节)
            dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
            # 使用 msg.point_step 跳过不需要的颜色/强度字段
            pc_data = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height, offset=0)
            
            # 提取 XYZ
            points = np.stack([pc_data['x'], pc_data['y'], pc_data['z']], axis=1)
            
            # 过滤 NaN 和 0 深度
            mask = (points[:, 2] > 0.01) & (~np.isnan(points).any(axis=1))
            final_points = points[mask]

            if len(final_points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(final_points.astype(np.float64))
                
                # 每一帧都保存，你可以随时 Ctrl+C 停止
                out_path = os.path.join(args.output_dir, f"frame_{count:04d}.pcd")
                o3d.io.write_point_cloud(out_path, pcd)
                count += 1
            
            if count % 10 == 0:
                print(f"已提取 {count} 帧点云...", end='\r')

    print(f"\n--- 调试结束：共提取 {count} 帧 ---")

if __name__ == "__main__":
    main()