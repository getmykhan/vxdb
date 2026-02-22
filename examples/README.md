# vxdb Examples

Step-by-step Jupyter notebooks showing how to use vxdb with popular embedding providers.

## Notebooks

| Notebook | Description | API Key? | Install |
|----------|-------------|----------|---------|
| **[quickstart.ipynb](quickstart.ipynb)** | All core features in 5 minutes (dummy vectors) | No | `pip install vxdb` |
| **[openai_embeddings.ipynb](openai_embeddings.ipynb)** | OpenAI `text-embedding-3-small` (1536-dim) | Yes | `pip install vxdb openai` |
| **[sentence_transformers.ipynb](sentence_transformers.ipynb)** | Local embeddings with Hugging Face (384-dim, free) | No | `pip install vxdb sentence-transformers` |
| **[langchain_integration.ipynb](langchain_integration.ipynb)** | LangChain with any provider + RAG pipeline | Depends | `pip install vxdb langchain-openai langchain-huggingface` |
| **[cohere_embeddings.ipynb](cohere_embeddings.ipynb)** | Cohere `embed-v4.0` with multilingual search | Yes | `pip install vxdb cohere` |
| **[hybrid_search.ipynb](hybrid_search.ipynb)** | Deep dive into hybrid search, alpha tuning | No | `pip install vxdb sentence-transformers` |

## Where to start

**New to vxdb?** Start with `quickstart.ipynb` — no API keys, runs immediately.

**Want real embeddings?**
- Free / local: `sentence_transformers.ipynb`
- Best quality (paid): `openai_embeddings.ipynb`
- Multilingual: `cohere_embeddings.ipynb`
- Provider-agnostic: `langchain_integration.ipynb`

**Want hybrid search?** `hybrid_search.ipynb` covers when to use it and how to tune `alpha`.

## Running the notebooks

```bash
# 1. Install vxdb
pip install vxdb

# 2. Install Jupyter
pip install jupyter

# 3. Run
jupyter notebook examples/
```

Or open them directly in VS Code, Cursor, or Google Colab.
