from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
TEXT_FILE = BASE_DIR / 'text' / 'text.txt'
THRESHOLD = 64


def main() -> None:
    if not TEXT_FILE.exists():
        raise FileNotFoundError(f'Khong tim thay file: {TEXT_FILE}')

    total_lines = 0
    matched_lines = 0
    matched_entries = []
    kept_lines = []

    with TEXT_FILE.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            total_lines += 1
            content = line.rstrip('\r\n')

            if len(content) >= THRESHOLD:
                matched_lines += 1
                matched_entries.append((line_number, len(content), content))
            else:
                kept_lines.append(line)

    ratio = (matched_lines / total_lines * 100.0) if total_lines else 0.0

    if matched_entries:
        print(f'Nhung dong co do dai >= {THRESHOLD} ky tu (se bi xoa):')
        for line_number, line_length, content in matched_entries:
            print(f'[{line_number}] ({line_length} ky tu): {content}')

        with TEXT_FILE.open('w', encoding='utf-8') as f:
            f.writelines(kept_lines)

        print('Da xoa cac dong tren khoi text.txt')
    else:
        print(f'Khong co dong nao co do dai >= {THRESHOLD} ky tu')

    print(f'File: {TEXT_FILE}')
    print(f'Tong so dong: {total_lines:,}')
    print(f'So dong co do dai >= {THRESHOLD} ky tu: {matched_lines:,}')
    print(f'Ty le: {ratio:.2f}%')
    print(f'So dong con lai sau khi xoa: {len(kept_lines):,}')


if __name__ == '__main__':
    main()
