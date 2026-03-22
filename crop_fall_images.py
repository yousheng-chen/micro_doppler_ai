import os
from PIL import Image
import numpy as np
from pathlib import Path


def crop_fall_images(
    input_dir='D:\Code\python\Micro-Doppler-vit\micro_doppler_ai\data\original_data\Wava',
    output_dir='D:\Code\python\Micro-Doppler-vit\micro_doppler_ai\data\cropped_data_448\Wava',
    crop_height=224,
    crop_width=224,
    stride=54
):
    """
    对Fall文件夹中的图像进行裁剪。
    从图像的中间提取224高度的区域，然后使用滑动窗口（步长112）提取224x224的块。
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出裁剪后图像的目录
        crop_height: 裁剪高度（从中间提取）
        crop_width: 裁剪宽度
        stride: 滑动窗口步长
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted([
        f for f in os.listdir(input_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if not image_files:
        print(f"错误: 在 {input_dir} 中未找到图像文件")
        return
    
    print(f"{'='*70}")
    print(f"开始处理Fall图像裁剪")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"裁剪参数 - 高度: {crop_height}, 宽度: {crop_width}, 步长: {stride}")
    print(f"{'='*70}\n")
    
    total_crops = 0
    
    # 处理每张图像
    for img_idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        
        try:
            # 打开图像
            img = Image.open(image_path)
            
            # 转换为RGB（如果需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 获取图像尺寸
            img_width, img_height = img.size
            
            print(f"[{img_idx}/{len(image_files)}] 处理: {image_file}")
            print(f"  原始尺寸: {img_width}x{img_height}")
            
            # 检查图像是否足够高
            if img_height < crop_height:
                print(f"  警告: 图像高度({img_height}) < 裁剪高度({crop_height})，跳过")
                continue
            
            # 计算中间区域的起始y坐标
            # 从图像中间提取crop_height高度的区域
            center_y = img_height // 2
            start_y = center_y - crop_height // 2
            
            # 确保裁剪区域在图像内
            if start_y < 0:
                start_y = 0
            if start_y + crop_height > img_height:
                start_y = img_height - crop_height
            
            # 从中间高度区域裁剪
            img_middle = img.crop((0, start_y, img_width, start_y + crop_height))
            middle_width = img_middle.width
            
            print(f"  中间高度区域: y范围 [{start_y}, {start_y + crop_height}]")
            print(f"  中间区域尺寸: {middle_width}x{crop_height}")
            
            # 使用步长进行滑动窗口裁剪
            crop_count = 0
            for x_pos in range(0, middle_width - crop_width + 1, stride):
                # 裁剪224x224块
                crop_box = (x_pos, 0, x_pos + crop_width, crop_height)
                cropped_img = img_middle.crop(crop_box)
                
                # 生成输出文件名 (原文件名_块序号)
                base_name = Path(image_file).stem
                output_filename = f"{base_name}_crop_{crop_count}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存裁剪后的图像
                cropped_img.save(output_path, 'JPEG', quality=95)
                crop_count += 1
                total_crops += 1
            
            print(f"  生成裁剪块数: {crop_count}")
            print()
            
        except Exception as e:
            print(f"  错误: 处理 {image_file} 失败 - {e}\n")
    
    print(f"{'='*70}")
    print(f"处理完成！")
    print(f"原始图像数: {len(image_files)}")
    print(f"总裁剪块数: {total_crops}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    crop_fall_images()
