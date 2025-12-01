import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 定义一个常数，作为局部亮度归一化的目标中值亮度
TARGET_MEDIAN_L = 100.0 # 与原代码保持一致

def crop_image(IMG_PATH: Path, OUTPUT_ROOT: Path): 
    # 遍历一级文件夹（浓度值）
    for folder in IMG_PATH.iterdir():
        if folder.is_dir():
            concentration = folder.name  # 保存原文件夹名（浓度值）
    
            # 创建对应输出文件夹
            output_dir = OUTPUT_ROOT / concentration
            output_dir.mkdir(parents=True, exist_ok=True)
    
            # 遍历该浓度文件夹下的所有图片
            for img_path in tqdm(folder.glob("*.*")):
                if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif"]:
                    continue  # 跳过非图像文件
    
                # 原文件名 + _crop 后缀
                new_name = img_path.stem + "_crop" + img_path.suffix
    
                # 输出文件路径
                output_file = output_dir / new_name
    
                # 执行 crop 函数（函数内部会保存）
                crop_and_save_circle_region(
                    img_path=img_path,
                    output_dir=output_file.parent,
                    output_size=128,
                    padding=10
                )
    
            print(f"Processed: {folder}")

def crop_and_save_circle_region(
    img_path: Path, 
    output_dir: Path, 
    output_size: int = 128, 
    padding: int = 10
) -> bool:
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = img_path.name
    output_path = output_dir / output_filename

    try:
        # 1. 加载图像
        img = Image.open(img_path).convert("RGB")
        orig_W, orig_H = img.size
        img_np_original_for_processing = np.array(img)
        
        # ======================================================
        # 2. 霍夫检测 (降采样检测 + 坐标还原)
        # ======================================================
        DETECTION_TARGET_DIM = 1024
        scale_factor = 1.0
        max_dim = max(orig_W, orig_H)
        
        gray_full_original = cv2.cvtColor(img_np_original_for_processing, cv2.COLOR_RGB2GRAY)
        detection_gray = gray_full_original
        
        if max_dim > DETECTION_TARGET_DIM:
            scale_factor = DETECTION_TARGET_DIM / max_dim
            new_W = int(orig_W * scale_factor)
            new_H = int(orig_H * scale_factor)
            detection_gray = cv2.resize(gray_full_original, (new_W, new_H), interpolation=cv2.INTER_AREA)
            
        gray_blur_small = cv2.GaussianBlur(detection_gray, (9, 9), 2)
        small_H, small_W = gray_blur_small.shape
        small_center = np.array([small_W / 2, small_H / 2])

        hough_dp = 2.0
        hough_param2 = 40

        circles = cv2.HoughCircles(
            gray_blur_small,
            cv2.HOUGH_GRADIENT,
            dp=hough_dp,
            minDist=small_H / 5,
            param1=100,
            param2=hough_param2,
            minRadius=int(20 * scale_factor),
            maxRadius=min(small_H, small_W) // 2,
        )

        best_circle_original_scale = None

        if circles is not None:
            circles_found = circles[0]
            best_score = -float('inf')
            best_circle_small = None

            # 寻找最接近中心的圆
            for (sx, sy, sr) in circles_found:
                dist_to_center = np.linalg.norm(np.array([sx, sy]) - small_center)
                score = -dist_to_center
                
                if score > best_score:
                    best_score = score
                    best_circle_small = (sx, sy, sr)
            
            if best_circle_small is not None:
                sx, sy, sr = best_circle_small
                ox = sx / scale_factor
                oy = sy / scale_factor
                or_ = sr / scale_factor
                best_circle_original_scale = (ox, oy, or_)
        
        # ======================================================
        # 3. 局部亮度归一化
        # ======================================================
        img_np_normalized = img_np_original_for_processing.copy()

        if best_circle_original_scale is not None:
            x, y, r = best_circle_original_scale
            x, y, r = int(round(x)), int(round(y)), int(round(r))

            mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            lab_original = cv2.cvtColor(img_np_original_for_processing, cv2.COLOR_RGB2LAB)
            l_channel_original, a_channel_original, b_channel_original = cv2.split(lab_original)

            l_in_circle = l_channel_original[mask > 0]
            
            if len(l_in_circle) > 0:
                current_median_l_in_circle = np.median(l_in_circle)
                # 计算亮度平移量
                shift_amount = TARGET_MEDIAN_L - current_median_l_in_circle

                l_shifted = l_channel_original.astype(np.float32) + shift_amount
                l_shifted = np.clip(l_shifted, 0, 255).astype(np.uint8)
                
                lab_normalized = cv2.merge([l_shifted, a_channel_original, b_channel_original])
                img_np_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
            
        img_processed_for_cropping = Image.fromarray(img_np_normalized)

        # ======================================================
        # 4. 裁剪、缩放并保存
        # ======================================================
        cropped_img_pil = None
        
        if best_circle_original_scale is not None:
            # 成功检测到圆点，进行精确裁剪
            x, y, r = best_circle_original_scale
            x, y, r = int(round(x)), int(round(y)), int(round(r))
            
            left = max(x - r - padding, 0)
            top = max(y - r - padding, 0)
            right = min(x + r + padding, orig_W)
            bottom = min(y + r + padding, orig_H)

            # 确保裁剪框有效
            if right > left and bottom > top:
                cropped_img_pil = img_processed_for_cropping.crop((left, top, right, bottom))
            else:
                print(f"警告: {output_filename} 计算的裁剪框无效。")
        
        if cropped_img_pil is None:
             # 未检测到圆点或裁剪框无效，进行兜底中心裁剪
            print(f"警告: {output_filename} 未检测到圆点或裁剪失败，进行中心裁剪。")
            w, h = img.size 
            crop_dim = min(w, h)
            left = (w - crop_dim) // 2
            top = (h - crop_dim) // 2
            right = left + crop_dim
            bottom = top + crop_dim
            cropped_img_pil = img.crop((left, top, right, bottom))
            

        # 最终缩放到目标尺寸并保存
        cropped_img_pil = cropped_img_pil.resize((output_size, output_size), Image.Resampling.LANCZOS)
        cropped_img_pil.save(output_path)

        return True

    except Exception as e:
        print(f"处理图像 {img_path.name} 时发生错误: {e}")
        return False