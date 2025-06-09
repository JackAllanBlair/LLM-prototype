from typing import List
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

""" 

load_documents.py serves to split up or 'chunk' various forms of data, returning a list of chunks

TODO: add in logic to check what kind of data is being parsed and switch between various chunking functions
TODO: add different chunking functions for different forms of data
TODO: add logic to throw exception if unsupported data is parsed into functions

"""

def chunk_text_by_pg(text):
    # create list for chunks
    chunks = []

    # initialize and configure text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,        # target chunk size in characters (or tokens)
        chunk_overlap = 50,      # overlap to maintain context between chunks
        separators = ["\n\n", "\n", " ", ""]  
        # The splitter will try to split by double newline (paragraph), then newline, then space, then as last resort character.
    )

    # split and return chunks
    chunks = text_splitter.split_text(text)
    return chunks

def load_and_chunk(file_path: str) -> List[str]:
    # get markdown text for all pages
    md_text = pymupdf4llm.to_markdown(file_path)

    # chunk data and return chunks
    chunks = chunk_text_by_pg(md_text) 
    return chunks

