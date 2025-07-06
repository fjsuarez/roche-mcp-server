# Roche MCP Server

A Model Context Protocol (MCP) server that exposes API endpoints as tools for LLMs to use with Ollama.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with llama3.2 model

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup the project

```bash
cd roche-mcp-server
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

### 3. Install Ollama and llama3.2

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull llama3.2 model
ollama pull llama3.2
```
## Usage

### 1. Start your API server

Make sure your API server is running on the configured URL (default: `http://127.0.0.1:8000`).

### 2. Start Ollama model

In a new terminal, start Ollama with llama3.2:

```bash
ollama run llama3.2
```
### 3. Start the MCP server

```bash
uv run api.py
```

Keep this running in a terminal.



## License

MIT