from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
from src.config import TXT_DIR


class CustomTextSplitter(TextSplitter):
    def __init__(self, separator: str = "[START]"):
        super().__init__()
        self.separator = separator

    # TODO: test this, some strange stuff happens
    def split_text(self, text: str) -> list[str]:
        chunks = text.split(self.separator)

        # LLM tends to add extra ``` at the beginning and end of file
        chunks[0] = chunks[0].replace("```", "")
        chunks[-1] = chunks[0].replace("```", "")
        
        # Del empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks
    
    def split_text_to_documents(self, text: str) -> list[Document]:
        chunks = self.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    
if __name__ == "__main__":
    test_file_path = TXT_DIR / "marked_chunks/code_du_travail_part_1_marked.txt"

    with open(test_file_path.as_posix(), 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = CustomTextSplitter()
    
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: \n {chunk}\n")

    print(f"Le nombre d'articles est: {len(chunks)}")
