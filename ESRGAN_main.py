import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import os
import time

def enhance_image(input_path, output_path):
    """
    Tăng độ phân giải ảnh với Real-ESRGAN
    """
    try:
        # Kiểm tra GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Sử dụng device: {device}')
        
        # Tạo model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        netscale = 4
        
        start_time = time.time()
        print("Đang tải model...")
        
        # Khởi tạo upsampler với URL trực tiếp và tiling
        upsampler = RealESRGANer(
            scale=netscale,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=512,  # Kích thước tile để xử lý từng phần
            tile_pad=32,
            pre_pad=0,
            half=False,
            device=device
        )
        
        # Đọc ảnh
        print("Image reading...")
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception(f"Can not read image: {input_path}")
        
        # Hiển thị kích thước ảnh
        h, w = img.shape[:2]
        print(f"Original image size: {w}x{h}")
        print(f"Processed image size will be: {w*4}x{h*4}")
        
        # Tăng độ phân giải
        print("Processing image... (may take a few minutes)")
        output, _ = upsampler.enhance(img, outscale=netscale)
        
        # Lưu ảnh
        print("Saving image...")
        cv2.imwrite(output_path, output)
        
        # Tính thời gian xử lý
        end_time = time.time()
        process_time = end_time - start_time
        print(f'Completed in {process_time:.2f} seconds')
        print(f'Image saved at: {output_path}')
        
    except Exception as e:
        print(f'Error: {str(e)}')

def main():
    # Tạo thư mục
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Kiểm tra xem có ảnh trong thư mục input không
    input_files = [f for f in os.listdir('input') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not input_files:
        print("No images found in the input folder. Please add images to the input folder.")
        return
    
    # Xử lý tất cả ảnh trong thư mục input
    total_files = len(input_files)
    for idx, file in enumerate(input_files, 1):
        print(f'\nProcessing image {idx}/{total_files}: {file}')
        print('-' * 50)
        input_path = os.path.join('input', file)
        output_path = os.path.join('output', f'enhanced_{file}')
        enhance_image(input_path, output_path)

if __name__ == '__main__':
    main()