from load_documents import load_and_chunk
from embed_documents import embed_chunks
from query_engine import query_rag
import os

"""

main.py serves as a basic test file to load documents, embed them, and test the model outputs in terminal

"""

# set to True if new data needs to be chunked and embedded
chunk_and_embed_data = True

if chunk_and_embed_data == True:
    # create list for all chunks
    all_chunks = []

    # simple loop to retrieve the name of each document in /documents
    for filename in os.listdir("..\\Documents"):

        # retrieve the path of each file in /documents
        file_path = os.path.join("..\\Documents", filename)

        # detect supported data types
        if os.path.isfile(file_path) and filename.lower().endswith((".pdf", ".txt")):

            # load and chunk each document
            chunks = load_and_chunk(file_path)

            # append to all_chunks list
            all_chunks.extend(chunks)
            print(file_path)

    # once all chunks are loaded, vectorize and save with embed_chunks function
    embed_chunks(all_chunks)

# simple look for testing model in terminal
while True:
    question = input("Ask your question: ")
    response = query_rag(question)
    print("Answer: ", response)

    if question.lower() == 'exit':
        break