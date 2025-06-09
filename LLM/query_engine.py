from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import os

"""
query_engine.py serves to initialize the tokenizer and model, create a prompt with context, and return an answer

TODO: add in 

"""

if os.path.exists("data/faiss_index"):
    # initialize embedding model
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # load vectorized data
    vectorstore = FAISS.load_local("data/faiss_index", embedder, allow_dangerous_deserialization=True)
else:
    print("No data in directory, continuing.")

# check if cuda is available, otherwise default to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# create and configure LLM
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
).to(device)
model.eval()

def query_rag(question: str) -> str:
    # retrieve relevant data
    docs = vectorstore.similarity_search(question, k=2)

    # build the structured context
    context = "\n\n".join(f"Chunk {i+1}: {doc.page_content}" for i, doc in enumerate(docs))

    # optimized prompt template
    prompt = f"""
    Answer the question using the context below. Cite page numbers and paragraphs where applicable.
    If the answer is not found in the context, say "I don't know based on the context provided."

    Context:
    {context}

    Question:
    {question}

    Answer:""".strip()

    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    # use autocast for mixed precision inference on GPU
    with torch.autocast(device_type='cuda', enabled=(device.type == "cuda")):
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract just the answer portion (after "Answer:")
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip().split("Question:")[0].strip()
    else:
        answer = response.strip()

    return answer