# Two-Tower Neural Network for Document Search

This project implements a two-tower neural network architecture for semantic document search, trained on the MS MARCO dataset.

## Quick Start

### Running with Python

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the server:

```bash
python src/main.py
```

The server will start on `http://0.0.0.0:8001`.

### Running with Docker

```bash
docker build -t two-tower-network .

# Run the container
docker run -p 8001:8001 two-tower-network
```

## API Usage

### Health Check

```bash
curl http://localhost:8001/health
```

### Document Search

```bash
# Search with default k=1
curl -X POST "http://localhost:8001/query" \
-H "Content-Type: application/json" \
-d '{"query": "how to make coffee"}'

# Search with custom k (e.g., k=3)
curl -X POST "http://localhost:8001/query" \
-H "Content-Type: application/json" \
-d '{"query": "how to make coffee", "k": 3}'
```

## Dataset Preprocessing

This model is trained on the [MS MARCO dataset](https://huggingface.co/datasets/microsoft/ms_marco). I wasn't able to push the data or faiss index due to size. To retrain the model or the index, you'll need to preprocess the dataset first:

1. Create training triplets:
   - For each query, extract the positive passage (is_selected=1) and randomly sample negative passages
   - Format: (query, positive passage text, negative passage text)
   - Save as `data/train_triplets.parquet` and `data/validation_triplets.parquet`

2. Create document index:
   - Extract all unique passages into a single file
   - Save as `data/unique_documents.parquet`

## Model Architecture

The system uses a two-tower architecture:

- Tower One: Encodes queries
- Tower Two: Encodes documents
- Both towers project inputs into the same embedding space
- FAISS is used for efficient similarity search
