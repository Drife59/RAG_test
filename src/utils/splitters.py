import os
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
from src.config import TXT_DIR
from pathlib import Path


# TODO: improve name
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
    
def get_each_article_as_unique_doc(splitter: TextSplitter, dir: Path) -> list[Document]:
    file_names = os.listdir(dir.as_posix())
    print("Number of marked files:", len(file_names))

    docs = []
    for file_name in file_names:
        with open(dir / Path(file_name), 'r', encoding='utf-8') as f:
            tagged_text = f.read()
            articles = splitter.split_text(tagged_text)

            for article in articles:
                docs.append(Document(page_content=article, metadata={"source": file_name}))
    
    return docs

if __name__ == "__main__":
    taggedfile_path = TXT_DIR / "tagged_chunks"
    splitter = CustomTextSplitter()
    docs = get_each_article_as_unique_doc(splitter, taggedfile_path)

    print(f"Il y a {len(docs)} articles dans le code du travail.")


def old_main():
    test_file_path = TXT_DIR / "tagged_chunks/code_du_travail_part_1_marked.txt"

    with open(test_file_path.as_posix(), 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = CustomTextSplitter()
    
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: \n {chunk}\n")

    print(f"Le nombre d'articles est: {len(chunks)}")
