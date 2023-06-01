from typing import Callable, List, Tuple, Dict
import re
import pdfplumber 
import PyPDF4

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_pages_from_pdf(pdf_file) -> List[Tuple[int, str]]:

    """
    Extract the text in the pages of the pdf file 

    """

    with pdfplumber.open(pdf_file) as pdf:
        pages = []

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            # check if it is not an empty page
            if text.strip():
                pages.append((page_num + 1, text))

    return pages


def extract_metadata_from_pdf(pdf_file) -> Dict:

    reader = PyPDF4.PdfFileReader(pdf_file)
    metadata = reader.getDocumentInfo()

    return {
        "title" : metadata.get("/Title", "").strip(),
        "author": metadata.get("/Author", "").strip(),
        "creation_date": metadata.get("/CreationDate", "").strip(),
    }


def parse_pdf(pdf_file) -> Tuple[List[Tuple[int, str]], Dict[str,str]]:

    metadata = extract_metadata_from_pdf(pdf_file)
    pages = extract_pages_from_pdf(pdf_file)
    
    return metadata, pages 

def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)



def clean_text(pages: List[Tuple[int, str]], 
                cleaning_functions: List[Callable[[str], str]]) -> List[Tuple[int, str]]:


    cleaned_pages = []

    for page_num, text in pages:

        for cleaning_function in cleaning_functions:
            text =  cleaning_function(text)

        cleaned_pages.append((page_num, text))

    return cleaned_pages

def create_chunks(text, metadata: Dict[str, str]) -> List[Document]:

    doc_chunks = []

    for page_num, page in text:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )

        chunks = text_splitter.split_text(page)

        for i, chunk in enumerate(chunks):

            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": 1, 
                    "source": f"p{page_num}-{i}",
                    **metadata,
                }
            )

            doc_chunks.append(doc)
    return doc_chunks