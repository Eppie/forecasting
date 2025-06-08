# Forecasting System

A sophisticated forecasting system that leverages Large Language Models (LLMs) and vector databases to generate predictions with supporting rationales and citations.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with LLM-based generation for informed forecasts
- **Vector Database Integration**: Uses Qdrant for efficient similarity search of relevant information
- **Document Processing**: Automated ingestion and chunking of source documents
- **LLM Integration**: Supports local LLM models via LlamaCpp
- **Logging**: Tracks forecast history and performance

## Prerequisites

- Python 3.13+
- PostgreSQL (for logging)
- Qdrant (vector database)
- LLM model file (compatible with LlamaCpp)

### System Dependencies (macOS Apple Silicon)

1. **Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **PostgreSQL** (via Homebrew):
   ```bash
   brew install postgresql@14
   brew services start postgresql@14
   ```

3. **Qdrant** (via Docker):
   ```bash
   # Install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

4. **Xcode Command Line Tools** (for building dependencies):
   ```bash
   xcode-select --install
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd forecasting
   ```

2. Install Python dependencies using `uv` (faster alternative to pip):
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate
   
   # Install dependencies
   uv pip install -e .
   ```

3. Set up environment variables in a `.env` file (see `.env.example` for reference)

## Configuration

Create a `.env` file in the project root with the following variables:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/forecastdb
LLM_MODEL_PATH=/path/to/your/model.gguf
EMBEDDING_MODEL_ID=all-MiniLM-L6-v2
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_COLLECTION_NAME=forecast_docs
```

## Usage

### Running the Forecast Pipeline

```python
from src.main import run_forecast_pipeline

result = run_forecast_pipeline("What is the probability of event X happening in the next month?")
print(f"Forecast: {result.probability_str}")
print(f"Rationale: {result.rationale}")
```

### CLI Usage

```bash
python -m forecasting.cli forecast "Your forecasting question here"
```

## Project Structure

```
forecasting/
├── src/
│   ├── forecasting/
│   │   ├── db/            # Database models and session management
│   │   ├── ingestion/     # Document ingestion and processing
│   │   ├── log/          # Forecast logging
│   │   ├── reasoning/    # LLM integration and prompt engineering
│   │   └── retrieval/    # Vector store and retrieval logic
│   └── main.py          # Main application entry point
├── tests/               # Test files
├── pyproject.toml       # Project metadata and dependencies
└── README.md           # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses `black` for code formatting and `isort` for import sorting:

```bash
# Format code
black .

# Sort imports
uv run isort .

# Run linter
uv run ruff check .
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.