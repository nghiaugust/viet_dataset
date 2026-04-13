import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

from tqdm import tqdm
from trdg.data_generator import FakeTextDataGenerator

# ==========================================
# MONKEY PATCH: SỬA LỖI PILLOW 10+ CHO TRDG
# ==========================================
import PIL.ImageFont
if not hasattr(PIL.ImageFont.FreeTypeFont, 'getsize'):
    def getsize_patch(self, text, *args, **kwargs):
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)
        width = right - left
        height = bottom - top
        # TRDG can underestimate Vietnamese glyph height with Pillow 10+.
        # Add a vertical safety pad so descenders/diacritics are not clipped.
        safety_pad = max(4, int(self.size * 0.35))
        return (width, height + safety_pad)
    PIL.ImageFont.FreeTypeFont.getsize = getsize_patch

if not hasattr(PIL.ImageFont.FreeTypeFont, 'getoffset'):
    def getoffset_patch(self, text, *args, **kwargs):
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)
        return (left, top)
    PIL.ImageFont.FreeTypeFont.getoffset = getoffset_patch
# ==========================================

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[1]
TEXT_FILE = BASE_DIR / 'text' / 'text.txt'
FONTS_DIR = BASE_DIR / 'fonts'
BGS_DIR = BASE_DIR / 'background_augmented'
OUT_DIR = BASE_DIR / 'dataset'

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

TOTAL_IMAGES = 2000000         # Tổng số ảnh cần tạo
BATCH_SIZE = 100000            # Số lượng ảnh tối đa trong 1 thư mục con
SEED = 42

# Data quality constraints
MIN_TEXT_LEN = 2
MAX_TEXT_LEN = 200             # TRDG tự cắt tối đa 200 ký tự/line
DEDUP_TEXT = True              # Bật để tránh text trùng lặp vô tình
CLEAN_OUTPUT_FIRST = True      # Xóa batch_* cũ trước khi tạo mới
RESUME_IF_POSSIBLE = True      # Tiếp tục chạy từ labels.txt nếu có
VALIDATE_FONTS = True          # Test nhanh font để giảm nguy cơ crash lúc sinh
STRICT_BACKGROUND_CHECK = True # Chặn file không phải ảnh trong thư mục background

# Font weighting (Times ~30%, font khác ~70%)
TARGET_FONTS = ['times.ttf', 'timesbd.ttf', 'timesbi.ttf', 'timesi.ttf']
TIMES_RATIO = 0.30

# Output files
LABEL_FILE = OUT_DIR / 'labels.txt'
REPORT_FILE = OUT_DIR / 'generation_report.json'

# ==========================================
# 2. HÀM TIỆN ÍCH
# ==========================================
def normalize_text(text: str) -> str:
    text = text.strip().replace('\t', ' ')
    # Chuẩn hóa khoảng trắng để tránh label bẩn
    return ' '.join(text.split())


def load_text_corpus(path: Path, dedup: bool = True, min_len: int = 1, max_len: int = 200):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file text: {path}")

    raw_lines = 0
    valid_lines = 0
    skipped_short = 0
    skipped_long = 0
    skipped_empty = 0
    texts = []

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            raw_lines += 1
            text = normalize_text(line)

            if not text:
                skipped_empty += 1
                continue

            if len(text) < min_len:
                skipped_short += 1
                continue

            if len(text) > max_len:
                skipped_long += 1
                continue

            texts.append(text)
            valid_lines += 1

    if dedup:
        before = len(texts)
        texts = list(dict.fromkeys(texts))
        removed_duplicates = before - len(texts)
    else:
        removed_duplicates = 0

    if not texts:
        raise ValueError("Không có dòng text hợp lệ sau khi lọc")

    stats = {
        'raw_lines': raw_lines,
        'valid_lines': valid_lines,
        'unique_texts': len(texts),
        'removed_duplicates': removed_duplicates,
        'skipped_empty': skipped_empty,
        'skipped_short': skipped_short,
        'skipped_long': skipped_long,
    }

    return texts, stats


def find_background_images(path: Path) -> Tuple[List[Path], List[Path]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục background: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Đường dẫn background không phải thư mục: {path}")

    image_files: List[Path] = []
    non_image_files: List[Path] = []

    for p in path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(p)
        else:
            non_image_files.append(p)

    image_files.sort()
    non_image_files.sort()
    return image_files, non_image_files


def build_weighted_fonts(fonts_dir: Path, target_fonts: Sequence[str], times_ratio: float = 0.30, seed: int = 42):
    if not fonts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục font: {fonts_dir}")
    if not fonts_dir.is_dir():
        raise NotADirectoryError(f"Đường dẫn font không phải thư mục: {fonts_dir}")

    all_font_paths = [str(p) for p in fonts_dir.rglob('*.ttf')]
    all_font_paths += [str(p) for p in fonts_dir.rglob('*.otf')]
    all_font_paths.sort()

    if not all_font_paths:
        raise ValueError("Không tìm thấy font nào trong thư mục fonts/")

    times_fonts = []
    other_fonts = []
    targets = {x.lower() for x in target_fonts}

    for f in all_font_paths:
        filename = os.path.basename(f).lower()
        if filename in targets:
            times_fonts.append(f)
        else:
            other_fonts.append(f)

    if other_fonts and times_fonts:
        # Tính số phần tử Times cần có để đạt ratio mong muốn
        # times/(times + others) = times_ratio
        target_times_count = int(len(other_fonts) * (times_ratio / max(1e-9, (1.0 - times_ratio))))
        multiplier = max(1, math.ceil(target_times_count / len(times_fonts)))
        weighted_times_fonts = times_fonts * multiplier
        final_font_paths = other_fonts + weighted_times_fonts
    else:
        weighted_times_fonts = times_fonts
        multiplier = 1
        final_font_paths = all_font_paths

    rng = random.Random(seed)
    rng.shuffle(final_font_paths)

    stats = {
        'all_fonts': len(all_font_paths),
        'times_fonts': len(times_fonts),
        'other_fonts': len(other_fonts),
        'weighted_fonts': len(final_font_paths),
        'times_multiplier': multiplier,
    }

    return final_font_paths, stats


def validate_fonts(font_paths: Sequence[str], size: int = 48):
    valid_fonts = []
    bad_fonts = []

    for font_path in tqdm(font_paths, desc='Kiem tra font', leave=False):
        try:
            font = PIL.ImageFont.truetype(font_path, size)
            font.getbbox('Tieng Viet OCR 123 aaAA')
            valid_fonts.append(font_path)
        except Exception as e:
            bad_fonts.append((font_path, str(e)))

    return valid_fonts, bad_fonts


def clean_output_dir(out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)

    for p in out_path.glob('batch_*'):
        if p.is_dir():
            shutil.rmtree(p)

    label_path = out_path / 'labels.txt'
    if label_path.exists():
        label_path.unlink()

    report_path = out_path / 'generation_report.json'
    if report_path.exists():
        report_path.unlink()


def save_report(report_path: Path, report: dict):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def pick_coprime_stride(length: int, seed: int) -> int:
    if length <= 1:
        return 1

    stride = (seed * 2 + 1) % length
    if stride == 0:
        stride = 1

    for _ in range(length):
        if math.gcd(stride, length) == 1:
            return stride
        stride = (stride + 2) % length
        if stride == 0:
            stride = 1

    return 1


def index_by_stride(i: int, length: int, stride: int, offset: int) -> int:
    return (offset + i * stride) % length


def scan_existing_labels(label_path: Path) -> Tuple[int, str]:
    if not label_path.exists():
        return 0, ''

    count = 0
    last_line = ''
    with label_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            last_line = line

    return count, last_line


def label_line_to_image_path(out_dir: Path, label_line: str) -> Path:
    rel_path = label_line.split('\t', 1)[0]
    rel_path = rel_path.replace('/', os.sep)
    return out_dir / rel_path


def main():
    print('=' * 80)
    print('SINH DATASET OCR CHAT LUONG CAO TU FILE TEXT')
    print('=' * 80)

    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUT_FIRST:
        print('Dang xoa du lieu batch_* cu de tranh tron dataset...')
        clean_output_dir(OUT_DIR)

    print(f"Doc text tu: {TEXT_FILE}")
    texts, text_stats = load_text_corpus(
        TEXT_FILE,
        dedup=DEDUP_TEXT,
        min_len=MIN_TEXT_LEN,
        max_len=MAX_TEXT_LEN,
    )

    generation_strings = texts
    text_stride = pick_coprime_stride(len(generation_strings), SEED + 13)
    text_offset = SEED % len(generation_strings)

    full_cycles = TOTAL_IMAGES // len(generation_strings)
    remainder = TOTAL_IMAGES % len(generation_strings)

    print(f"Tong so text hop le: {text_stats['valid_lines']:,}")
    print(f"So text unique dung de sinh: {len(generation_strings):,}")
    if DEDUP_TEXT:
        print(f"Da loai bo text trung: {text_stats['removed_duplicates']:,}")
    print(f"Ke hoach lap lai text: {full_cycles} vong + {remainder:,} mau")

    bg_images, non_image_bg_files = find_background_images(BGS_DIR)
    if not bg_images:
        raise ValueError(f"Khong tim thay anh nen trong: {BGS_DIR}")
    if non_image_bg_files and STRICT_BACKGROUND_CHECK:
        sample = ', '.join(x.name for x in non_image_bg_files[:5])
        raise ValueError(
            'Thu muc background co file khong phai anh o top-level, '
            'TRDG co the crash khi random trung file nay. '
            f'Vi du: {sample}'
        )
    print(f"So anh nen top-level tim thay: {len(bg_images):,}")

    final_font_paths, font_stats = build_weighted_fonts(
        FONTS_DIR,
        target_fonts=TARGET_FONTS,
        times_ratio=TIMES_RATIO,
        seed=SEED,
    )

    bad_fonts = []
    if VALIDATE_FONTS:
        print('Dang test nhanh font de loai bo font loi...')
        final_font_paths, bad_fonts = validate_fonts(final_font_paths, size=48)
        if not final_font_paths:
            raise ValueError('Tat ca font deu loi sau khi validate')

    font_stride = pick_coprime_stride(len(final_font_paths), SEED + 29)
    font_offset = (SEED * 7) % len(final_font_paths)

    print(f"Tong font tim thay: {font_stats['all_fonts']:,}")
    print(f"Font Times: {font_stats['times_fonts']:,} | Font khac: {font_stats['other_fonts']:,}")
    print(f"So phan tu font sau weighting: {font_stats['weighted_fonts']:,}")
    if VALIDATE_FONTS:
        print(f"Font loi bi loai: {len(bad_fonts):,}")

    start_index = 0
    label_mode = 'w'
    if RESUME_IF_POSSIBLE and not CLEAN_OUTPUT_FIRST:
        existing_labels, last_label_line = scan_existing_labels(LABEL_FILE)
        if existing_labels > 0:
            start_index = existing_labels
            label_mode = 'a'
            last_img = label_line_to_image_path(OUT_DIR, last_label_line)
            if not last_img.exists():
                raise RuntimeError(
                    'labels.txt dang ton tai nhung anh cuoi khong ton tai, '
                    f'khong the resume an toan: {last_img}'
                )

    if start_index >= TOTAL_IMAGES:
        print('Dataset da du so luong muc tieu, khong can sinh them.')
        return

    report = {
        'text_file': str(TEXT_FILE),
        'fonts_dir': str(FONTS_DIR),
        'background_dir': str(BGS_DIR),
        'output_dir': str(OUT_DIR),
        'total_images': TOTAL_IMAGES,
        'batch_size': BATCH_SIZE,
        'seed': SEED,
        'text_stats': text_stats,
        'font_stats': font_stats,
        'background_count_top_level': len(bg_images),
        'non_image_background_files': len(non_image_bg_files),
        'generator_strings': len(generation_strings),
        'expected_full_cycles': full_cycles,
        'expected_remainder': remainder,
        'start_index': start_index,
        'resume_enabled': RESUME_IF_POSSIBLE,
        'text_stride': text_stride,
        'text_offset': text_offset,
        'font_stride': font_stride,
        'font_offset': font_offset,
        'valid_fonts_after_validation': len(final_font_paths),
        'invalid_fonts_removed': len(bad_fonts),
        'created_at_utc': datetime.utcnow().isoformat() + 'Z',
        'notes': [
            'Duong dan duoc resolve theo vi tri script, khong phu thuoc cwd.',
            'Text va font duoc chon boi 2 chu ky doc lap de giam pattern modulo co dinh.',
            'Resume mode dua vao so dong labels.txt va kiem tra anh cuoi cung.',
        ],
    }
    save_report(REPORT_FILE, report)
    print(f"Da ghi bao cao thong ke: {REPORT_FILE}")

    print(f"Bat dau tao {TOTAL_IMAGES - start_index:,} anh (tu mau thu {start_index + 1:,})...")
    current_batch_name = None

    with LABEL_FILE.open(label_mode, encoding='utf-8') as f_label:
        for i in tqdm(range(start_index, TOTAL_IMAGES), total=TOTAL_IMAGES, initial=start_index, desc='Dang tao du lieu'):
            text_idx = index_by_stride(i, len(generation_strings), text_stride, text_offset)
            font_idx = index_by_stride(i, len(final_font_paths), font_stride, font_offset)
            text_for_image = generation_strings[text_idx]
            font_for_image = final_font_paths[font_idx]

            try:
                img = FakeTextDataGenerator.generate(
                    index=i + 1,
                    text=text_for_image,
                    font=font_for_image,
                    out_dir=None,
                    size=48,
                    extension=None,
                    skewing_angle=2,
                    random_skew=True,
                    blur=2,
                    random_blur=True,
                    background_type=3,
                    distorsion_type=1,
                    distorsion_orientation=1,
                    is_handwritten=False,
                    name_format=0,
                    width=-1,
                    alignment=1,
                    text_color='#000000,#1a1a1a,#2b2b2b,#3a3a3a',
                    orientation=0,
                    space_width=1.0,
                    character_spacing=1,
                    margins=(6, 10, 12, 10),
                    fit=False,
                    output_mask=False,
                    word_split=False,
                    image_dir=str(BGS_DIR),
                    stroke_width=0,
                    stroke_fill='#111111',
                    image_mode='RGB',
                    output_bboxes=0,
                )
            except Exception as e:
                raise RuntimeError(
                    f'Loi render tai index={i + 1}, text_idx={text_idx}, '
                    f'font={Path(font_for_image).name}'
                ) from e

            batch_num = (i // BATCH_SIZE) + 1
            batch_folder_name = f"batch_{batch_num:03d}"
            batch_path = OUT_DIR / batch_folder_name

            if batch_folder_name != current_batch_name:
                batch_path.mkdir(parents=True, exist_ok=True)
                current_batch_name = batch_folder_name

            img_filename = f"img_{i + 1:07d}.jpg"
            img_filepath = batch_path / img_filename
            img.save(img_filepath, format='JPEG', quality=95)

            safe_label = normalize_text(text_for_image)
            label_line = f"{batch_folder_name}/{img_filename}\t{safe_label}\n"
            f_label.write(label_line)

            if (i + 1) % 10000 == 0:
                f_label.flush()

    report['finished_at_utc'] = datetime.utcnow().isoformat() + 'Z'
    report['completed_images'] = TOTAL_IMAGES
    save_report(REPORT_FILE, report)

    print('\nHOAN THANH!')
    print(f"Anh + label duoc luu tai: {OUT_DIR}")
    print(f"Thong ke generation: {REPORT_FILE}")


if __name__ == '__main__':
    main()