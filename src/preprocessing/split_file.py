"""
This module split a big file into smaller chunks.
This prepare work for th extractor.
"""

from langchain_text_splitters import CharacterTextSplitter

from src.config import TXT_DIR

# This dir contains the "code du travail", splitted in "big" raw chunks.
RAW_CHUNK_PATH = TXT_DIR / "chunks"

def split_big_txtfile_in_chunks(input_file: str, output_prefix: str, chunk_size=10000, overlap=100):
    """
        Split a big text file in chunks.
        This are not chunks as LLM chunks, but just text chunks.
    """
    print(f"Splitting file {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        output_file_name = f"{output_prefix}_part_{i + 1}.txt"
        output_file_path = (RAW_CHUNK_PATH / output_file_name).as_posix()
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            f_out.write(chunk)
        print(f"File {output_file_name} created.")

    print(f"File {input_file} splitted. {len(chunks)} chunks created.")

if __name__ == "__main__":
    file_name = TXT_DIR / "code_du_travail_7mo.pdf.txt"
    # ~800 lines, ~7500 words, ~51K characters
    # This end up with 149 chunks
    split_big_txtfile_in_chunks(file_name.as_posix(), "code_du_travail", chunk_size=20000, overlap=1000)
