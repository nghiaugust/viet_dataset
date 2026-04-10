import os
import glob
import subprocess
from tqdm import tqdm

# Cấu hình đường dẫn
FONTS_DIR = 'fonts/'
BAD_FONTS_DIR = 'bad_fonts/' # Thư mục chứa các font bị hỏng để cách ly

# Tạo thư mục cách ly nếu chưa có
os.makedirs(BAD_FONTS_DIR, exist_ok=True)

# Quét toàn bộ font
fonts = glob.glob(os.path.join(FONTS_DIR, '*.ttf')) + glob.glob(os.path.join(FONTS_DIR, '*.otf'))

if not fonts:
    print("Không tìm thấy font nào trong thư mục fonts/!")
    exit()

print(f"Bắt đầu đưa {len(fonts)} fonts vào phòng thí nghiệm...\n")
bad_count = 0

for font_path in tqdm(fonts, desc="Đang test font"):
    # Đổi dấu slash để Windows không bị lỗi chuỗi khi truyền vào subprocess
    safe_path = font_path.replace('\\', '/')
    
    # Lệnh Python nhỏ gọn để vẽ thử chữ. Nếu font hỏng, lệnh này sẽ sập.
    test_code = (
        "from PIL import Image, ImageFont, ImageDraw; "
        "img = Image.new('RGB', (100, 50)); "
        "draw = ImageDraw.Draw(img); "
        f"font = ImageFont.truetype('{safe_path}', 20); "
        "draw.text((10, 10), 'Test', font=font)"
    )
    
    cmd = ["python", "-c", test_code]

    # Chạy tiến trình con và đợi kết quả (không in ra màn hình để tránh rác log)
    result = subprocess.run(cmd, capture_output=True, text=True)

    # returncode != 0 nghĩa là tiến trình con đã bị crash hoặc báo lỗi
    if result.returncode != 0:
        bad_count += 1
        print(f"\n[PHÁT HIỆN LỖI] Đã bắt được font hỏng: {os.path.basename(font_path)}")
        
        # Di chuyển file lỗi ra khỏi thư mục fonts gốc
        bad_dest = os.path.join(BAD_FONTS_DIR, os.path.basename(font_path))
        try:
            # Nếu file đã tồn tại ở đích thì xóa đi trước
            if os.path.exists(bad_dest):
                os.remove(bad_dest)
            os.rename(font_path, bad_dest)
            print(f"--> Đã cách ly thành công vào thư mục {BAD_FONTS_DIR}")
        except Exception as e:
            print(f"--> Không thể cách ly file này, vui lòng xóa bằng tay: {e}")

print("\n" + "="*50)
if bad_count > 0:
    print(f"KẾT LUẬN: Đã tiêu diệt và cách ly {bad_count} font hỏng.")
    print("Thư mục fonts/ của bạn hiện tại đã hoàn toàn sạch sẽ!")
else:
    print("KẾT LUẬN: Tuyệt vời! Không có font nào bị hỏng.")
print("="*50)