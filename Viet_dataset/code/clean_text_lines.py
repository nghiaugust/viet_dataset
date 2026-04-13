from pathlib import Path
import unicodedata


VOCAB_CHARS = "aAàÀảẢãÃáÁạẠăĂằĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 "
VOCAB_SET = set(VOCAB_CHARS)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def get_invalid_chars(text: str) -> list[str]:
    return sorted({ch for ch in text if ch not in VOCAB_SET})


def clean_file(file_path: Path) -> int:
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)

    kept_lines = []
    removed_count = 0

    for line in lines:
        line_content = normalize_text(line.rstrip("\r\n"))
        invalid_chars = get_invalid_chars(line_content)
        if invalid_chars:
            invalid_desc = ", ".join(f"U+{ord(ch):04X}('{ch}')" for ch in invalid_chars)
            print(f"Xoa dong: {line_content}")
            print(f"  Ky tu ngoai vocab: {invalid_desc}")
            removed_count += 1
        else:
            line_ending = ""
            if line.endswith("\r\n"):
                line_ending = "\r\n"
            elif line.endswith("\n"):
                line_ending = "\n"
            kept_lines.append(line_content + line_ending)

    file_path.write_text("".join(kept_lines), encoding="utf-8")
    print(f"Tong dong da xoa: {removed_count}")
    return removed_count


if __name__ == "__main__":
    target_file = Path(__file__).resolve().parents[1] / "text" / "text.txt"
    clean_file(target_file)