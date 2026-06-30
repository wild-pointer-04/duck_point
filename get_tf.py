import sys
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def extract_static_tf(bag_path):
    print(f"正在读取 {bag_path} 中的 TF 静态树...\n")
    typestore = get_typestore(Stores.LATEST)
    
    try:
        with Reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/tf_static':
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    for transform in msg.transforms:
                        parent = transform.header.frame_id
                        child = transform.child_frame_id
                        # 只筛选包含 camera 相关的 frame，避免打印太多无关信息
                        if 'camera' in parent or 'camera' in child:
                            print(f"父坐标系 (Parent): {parent}")
                            print(f"子坐标系 (Child):  {child}")
                            t = transform.transform.translation
                            r = transform.transform.rotation
                            print(f"平移 (Translation): x={t.x:.5f}, y={t.y:.5f}, z={t.z:.5f}")
                            print(f"旋转 (Rotation四元数): x={r.x:.5f}, y={r.y:.5f}, z={r.z:.5f}, w={r.w:.5f}")
                            print("-" * 40)
                    return  # tf_static 每帧都一样，读到一帧就退出
    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供 bag 路径！用法: python get_tf.py <bag_path>")
    else:
        extract_static_tf(sys.argv[1])