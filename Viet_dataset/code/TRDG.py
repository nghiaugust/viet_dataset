import os
import glob
from tqdm import tqdm
from trdg.generators import GeneratorFromDict

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
# Giả sử bạn chạy lệnh python từ thư mục gốc (chứa các thư mục background, text, fonts...)
TEXT_FILE = 'text/text.txt'
FONTS_DIR = 'fonts/'
BGS_DIR = 'background_augmented/'
OUT_DIR = 'dataset/'           # Thư mục tổng sẽ được tự động tạo ra
LABEL_FILE = os.path.join(OUT_DIR, 'labels.txt')

TOTAL_IMAGES = 2000000         # Tổng số ảnh cần tạo (2 triệu)
BATCH_SIZE = 100000            # Số lượng ảnh tối đa trong 1 thư mục con (để tránh treo ổ cứng)

# ==========================================
# 2. CHUẨN BỊ MÔI TRƯỜNG VÀ TỶ LỆ FONT
# ==========================================
os.makedirs(OUT_DIR, exist_ok=True)

# Quét tất cả các file font
all_font_paths = glob.glob(os.path.join(FONTS_DIR, '*.ttf')) + glob.glob(os.path.join(FONTS_DIR, '*.otf'))
if not all_font_paths:
    raise ValueError("Không tìm thấy font nào trong thư mục fonts/!")

# Tách riêng 4 font Times và các font còn lại
times_fonts = []
other_fonts = []

# Danh sách tên các font bạn muốn ép tỷ lệ (viết thường để dễ so sánh)
target_fonts = ['times.ttf', 'timesbd.ttf', 'timesbi.ttf', 'timesi.ttf']

for f in all_font_paths:
    filename = os.path.basename(f).lower() # Lấy tên file và chuyển thành chữ thường
    if filename in target_fonts:
        times_fonts.append(f)
    else:
        other_fonts.append(f)

# --- THUẬT TOÁN ÉP TỶ LỆ 30/70 ---
# Nếu other_fonts chiếm 70%, ta tính xem số lượng phần tử cần thiết cho 30% là bao nhiêu
if len(other_fonts) > 0 and len(times_fonts) > 0:
    target_times_count = int(len(other_fonts) * (30.0 / 70.0))
    
    # Tính số lần cần nhân bản 4 file Times hiện có
    multiplier = max(1, target_times_count // len(times_fonts))
    
    # Nhân bản danh sách font Times
    weighted_times_fonts = times_fonts * multiplier
    
    # Gom lại thành danh sách cuối cùng truyền vào TRDG
    final_font_paths = other_fonts + weighted_times_fonts
    
    print(f"Đã cấu hình tỷ lệ font: Font thường ({len(other_fonts)} file) chiếm 70%.")
    print(f"Font Times được nhân bản {multiplier} lần (tổng {len(weighted_times_fonts)} phần tử) để chiếm 30%.")
else:
    # Fallback nếu bạn chỉ có toàn font Times hoặc không có font Times nào
    final_font_paths = all_font_paths

# Đảm bảo file labels.txt được tạo mới
if os.path.exists(LABEL_FILE):
    os.remove(LABEL_FILE)

# ==========================================
# 3. KHỞI TẠO GENERATOR (BỘ SINH ẢNH)
# ==========================================
print(f"Đang nạp {len(final_font_paths)} fonts và cấu hình TRDG...")

generator = GeneratorFromDict(
    count=TOTAL_IMAGES,
    path=TEXT_FILE,
    fonts=final_font_paths,
    language='vn',
    size=32,                   # Kích thước font chữ cơ bản
    skewing_angle=3,           # Góc nghiêng tối đa (độ) - để nhỏ thôi cho giống chụp cam
    random_skew=True,          # Bật nghiêng ngẫu nhiên
    blur=1,                    # Độ mờ
    random_blur=True,          # Bật làm mờ ngẫu nhiên (chống out-focus)
    background_type=1,         # 1 = Sử dụng ảnh từ thư mục (image_dir)
    image_dir=BGS_DIR,         # Đường dẫn tới thư mục chứa nền nhiễu, giấy tờ
    distorsion_type=3,         # 3 = Bóp méo ngẫu nhiên (giả lập giấy nhăn/cong)
    text_color='#000000,#2b2b2b,#1a1a1a', # Đa dạng hóa màu mực (đen sậm, xám đậm)
    fit=True                   # Cắt background vừa khít với chữ
)

# ==========================================
# 4. VÒNG LẶP SINH ẢNH VÀ GHI NHÃN
# ==========================================
print(f"Bắt đầu gen {TOTAL_IMAGES} ảnh...")

# Mở file nhãn ở chế độ 'a' (append - ghi nối tiếp)
with open(LABEL_FILE, 'a', encoding='utf-8') as f_label:
    
    # Dùng tqdm để tạo thanh tiến trình hiển thị % hoàn thành
    for i, (img, lbl) in enumerate(tqdm(generator, total=TOTAL_IMAGES, desc="Đang tạo dữ liệu")):
        
        # --- Logic chia Batch ---
        # Tính toán xem ảnh hiện tại thuộc batch số mấy (bắt đầu từ 1)
        batch_num = (i // BATCH_SIZE) + 1
        batch_folder_name = f"batch_{batch_num:03d}" # VD: batch_001, batch_002
        batch_path = os.path.join(OUT_DIR, batch_folder_name)
        
        # Nếu là ảnh đầu tiên của batch mới, tạo thư mục batch đó
        if i % BATCH_SIZE == 0:
            os.makedirs(batch_path, exist_ok=True)
            
        # --- Lưu ảnh ---
        # Tên file ảnh định dạng: img_0000001.jpg
        img_filename = f"img_{i+1:07d}.jpg"
        img_filepath = os.path.join(batch_path, img_filename)
        
        # Lưu ảnh với chất lượng 95% (có thể giảm xuống 85-90 nếu muốn tiết kiệm dung lượng)
        img.save(img_filepath, format='JPEG', quality=95)
        
        # --- Ghi nhãn ---
        # Định dạng chuẩn: batch_001/img_0000001.jpg \t Nội dung text
        # CỰC KỲ QUAN TRỌNG: Phải dùng \t (tab) để ngăn cách
        label_line = f"{batch_folder_name}/{img_filename}\t{lbl}\n"
        f_label.write(label_line)

print("\nHOÀN THÀNH! Toàn bộ dữ liệu đã được lưu tại thư mục:", OUT_DIR)