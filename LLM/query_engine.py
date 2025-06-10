from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import os

"""
query_engine.py serves to initialize the tokenizer and model, create a prompt with context, and return an answer

"""

if os.path.exists("data/faiss_index"):
    # initialize embedding model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # load vectorized data
    global vectorstore
    vectorstore = FAISS.load_local("data/faiss_index", embedder, allow_dangerous_deserialization=True)
else:
    print("No data in directory, continuing.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    device="cuda:0",
    use_safetensors=True,
    trust_remote_code=True,
)

def query_rag(question: str):
    # retrieve relevant data
    docs = vectorstore.similarity_search(question, k=5)

    # build the structured context
    context = "\n\n".join(f"Chunk {i+1}: {doc.page_content}" for i, doc in enumerate(docs))

    # optimized prompt template
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=200)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


    # extract just the answer portion (after "Answer:")
    if "Answer:" in decoded_output:
        answer = decoded_output.split("Answer:")[-1].strip().split("Question:")[0].strip()
    else:
        answer = decoded_output.strip()

    return answer, context