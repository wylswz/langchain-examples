# Using the example

Install `ollama` and dependencies using `uv sync` and evaluate the notebook.

Add `.pdf` files to `library/` directory as knowledge base.

To visualize all graphs
```
pip install --upgrade "langgraph-cli[inmem]"
langgraph dev
```

# Vector store
this project uses Qdrant as vector store.

# Binary deps
```
# https://github.com/UB-Mannheim/tesseract/wiki
# for OCR
Tesseract
```
