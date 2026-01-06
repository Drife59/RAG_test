import glob
from pathlib import Path

import pypdf

from src.config import PDF_DIR, TXT_DIR


def read_pdf(file_path: str) -> str:
    """Read a PDF file and extract text content."""
    pdf_reader = pypdf.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def _write_text_file(content: str, pdf_file_name: str) -> None:
    file_destination = TXT_DIR / pdf_file_name

    with open(f"{file_destination}.txt", "w", encoding="utf-8") as file:
        file.write(content)
        print(f'Document "{pdf_file_name}" has been written as txt and contain {len(content)} characters.')


def _read_pdf_and_write_text_file(pdf_file: Path) -> None:
    pdf_file_name = pdf_file.name
    content = read_pdf(pdf_file.as_posix())
    _write_text_file(content, pdf_file_name)


def fetch_pdfs_and_save_as_raw(dir: Path) -> None:
    print(f"✓ Fetching documents from dir {dir}")
    pdf_files = glob.glob(dir.as_posix() + "/*.pdf")

    print(f"Found following pdf files: {pdf_files}")

    for pdf_file in pdf_files:
        _read_pdf_and_write_text_file(Path(pdf_file))


def fetch_pdf_and_save_as_raw(pdf_file: Path) -> None:
    print(f"✓ Fetching file {pdf_file}")
    _read_pdf_and_write_text_file(pdf_file)


if __name__ == "__main__":
    print("Running pdf discovery and conversion to txt...")
    print(f"fetching pdfs from PDF_DIR: {PDF_DIR}")
    fetch_pdfs_and_save_as_raw(PDF_DIR)
    # fetch_pdf_and_save_as_raw(CODE_DU_TRAVAIL_PATH)
