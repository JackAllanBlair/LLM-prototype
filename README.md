# private_gpt_prototype

Below is a starting outline of all the various components that may be implemented in this project   

| Component | Tool / Stack | Notes |
| --------- | ------------ | ----- |
OS	|	Windows	|		Could be any OS
LLM	|	Mistral 7B, Phi-2, TinyLLaMA, MythoMax	|		Run 4-bit quantized with 6GB VRAM
Vector DB |	Chroma or FAISS					|	Light, simple, no GPU needed
Embeddings |	all-MiniLM or bge-small-en via sentence-transformers |	Efficient and accurate
RAG Engine |	LangChain or LlamaIndex			|		Handles chunking, vectorization, retrieval
Web UI	|	Streamlit, Gradio, or FastAPI + HTML	|		Lightweight, interactive
File Input	| PDFs, TXT, DOCX (via unstructured or PyMuPDF)	|	PDFs now, other formats later
Authentication |	Basic auth or token-gated interface		|	Lightweight for PoC
Languages |	Optional LibreTranslate API (hosted or local)	|	Skip or stub for PoC
Monitoring |	Console logs / Prometheus + Grafana (optional)	|	Low priority for PoC