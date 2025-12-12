import pypdf

CODE_DU_TRAVAIL_NAME = "code_du_travail.pdf"
CODE_DU_TRAVAIL_PATH = "./text_sources/" + CODE_DU_TRAVAIL_NAME

SYNTEC_CONVENTION_NAME = "convention_collective_syntec.pdf"
SYNTEC_CONVENTION_PATH = "./text_sources/" + SYNTEC_CONVENTION_NAME

def read_pdf(file_path: str) -> str:
    """Read a PDF file and extract text content."""
    pdf_reader = pypdf.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


if __name__ == "__main__":
    code_du_travail_test = read_pdf(CODE_DU_TRAVAIL_PATH)
    print(f'La taille totale du code_du_travail_est: {len(code_du_travail_test)}')

    convention_syntec_test = read_pdf(SYNTEC_CONVENTION_PATH)
    print(f'La taille totale de la convention_syntec_est: {len(convention_syntec_test)}')