import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def main():
    bag_path = "duck_1_0.db3" # 确认文件名正确
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: get_message(t.type) for t in topic_types}
    
    # 我们看深度相机的内参，因为点云是基于深度图生成的
    target_topic = '/camera/depth/camera_info'

    print(f"正在从 {bag_path} 中读取内参...")
    
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == target_topic:
            msg = deserialize_message(data, type_map[target_topic])
            
            print("\n--- 相机内参 (Intrinsic Matrix K) ---")
            k = msg.k
            print(f"fx: {k[0]}")
            print(f"fy: {k[4]}")
            print(f"cx: {k[2]}")
            print(f"cy: {k[5]}")
            
            print("\n--- 畸变参数 (Distortion D) ---")
            print(f"D: {msg.d}")
            
            print("\n--- 图像分辨率 ---")
            print(f"Width: {msg.width}")
            print(f"Height: {msg.height}")
            break # 只看一帧就够了

if __name__ == "__main__":
    main()