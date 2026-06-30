import open3d as o3d
import numpy as np
import os
import glob

def process_one_frame(input_file, output_file):
    pcd = o3d.io.read_point_cloud(input_file)
    if pcd.is_empty():
        return False

    # --- 1. 精准裁剪范围 (基于你刚才成功的定位) ---
    # 稍微放大一点点，确保不切掉鸭子屁股
    min_bound = np.array([-0.20, -0.20, 0.45])
    max_bound = np.array([ 0.20,  0.20, 1.00])
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # 这里的变量名统一为 cropped
    cropped = pcd.crop(bbox)

    if cropped.is_empty():
        return False

    # --- 2. 移除桌面平面 (RANSAC) ---
    # distance_threshold 是点到平面的距离阈值，0.01米即1厘米
    try:
        plane_model, inliers = cropped.segment_plane(
            distance_threshold=0.015, 
            ransac_n=3, 
            num_iterations=1000
        )
        # invert=True 表示删除平面，保留鸭子
        duck_only = cropped.select_by_index(inliers, invert=True)
    except:
        duck_only = cropped

    # --- 3. 统计去噪 (清理周围散乱的点) ---
    final_pcd, _ = duck_only.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if not final_pcd.is_empty():
        o3d.io.write_point_cloud(output_file, final_pcd)
        return True
    return False

def main():
    input_dir = "raw_pcd_data"
    output_dir = "duck_only"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcd_files = sorted(glob.glob(os.path.join(input_dir, "*.pcd")))
    print(f"开始批量处理 {len(pcd_files)} 个文件...")

    count = 0
    for f in pcd_files:
        out_path = os.path.join(output_dir, "clean_" + os.path.basename(f))
        if process_one_frame(f, out_path):
            count += 1
            if count % 50 == 0:
                print(f"已成功提取 {count} 帧鸭子...")

    print(f"\n大功告成！共保存 {count} 帧干净的点云到 {output_dir}")

if __name__ == "__main__":
    main()