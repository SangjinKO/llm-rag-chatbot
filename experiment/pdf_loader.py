from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class PageText:
    page_num: int
    text: str


def extract_pdf_text(pdf_path: Path) -> List[PageText]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: List[PageText] = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        
        # Clean up the format (remove spaces)
        cleaned = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
        
        pages.append(PageText(page_num=i, text=cleaned))

    return pages


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # PDF_PATH
    PDF_PATH = project_root / "data" / "sample_en_short.pdf"
    out_path = project_root / "output" / "text_dump.txt"

    pages = extract_pdf_text(PDF_PATH)

    # Text extraction summary
    total_chars = sum(len(p.text) for p in pages)
    print(f"[OK] Loaded: {PDF_PATH.name}")
    print(f"[OK] Pages: {len(pages)} | Total chars: {total_chars}")

    # Save to a TXT file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n===== PAGE {p.page_num} =====\n")
            f.write(p.text + "\n")

    # Print the initial part
    if pages:
        snippet = pages[0].text[:600]
        print("\n--- First page preview (first 600 chars) ---")
        print(snippet if snippet else "(No extractable text on page 1)")
        print("\n[OK] Saved to:", out_path)

if __name__ == "__main__":
    main()
