# Practical Introduction to AI Agents for Field Use - Chapter 4

This directory contains source code and resources related to Chapter 4 of the book "Practical Introduction to AI Agents for Field Use" (Kodansha).

To run the code in Chapter 4, follow the steps below.

## Prerequisites

To run this project, you need the following:

- Python 3.12 or higher
- Docker and Docker Compose
- VS Code
- Opened as a workspace using VS Code's Multi-root Workspaces feature (see [here](../README.md) for instructions)
- OpenAI account and API key

Python dependencies are listed in `pyproject.toml`.

## Environment Setup

### 1. Open the chapter 4 workspace
Create a virtual environment in the chapter 4 directory.
Select `chapter 4` in the Add Terminal in VS Code.

### 2. Install uv

Use `uv` to resolve dependencies.
If you have never used `uv`, install it as follows:

Using `pip`:
```bash
pip install uv
```

For Mac or Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create a Python Virtual Environment and Install Dependencies

Install Dependencies
```bash
uv sync
```

Activate the virtual environment created after installation.

``bash
source .venv/bin/activate
```

### 4. Set Environment Variables
Create a `.env` file and add the following content:

If you do not have an OpenAI API key, obtain one from the [OpenAI official website](https://platform.openai.com/).

```bash

```env
# OpenAI API Settings
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_MODEL= "gpt-4o-2024-08-06"
```

### 5. Building a Search Index

Use the make command.

```bash
#Starting the Container
make start.engine

#Building the Index
make create.index
```

If an error occurs in the Elasticsearch container when running `create.index`, comment out the following line in `docker-compose.yml`.
If you comment it out, Elasticsearch data will not be persisted, so you will need to rebuild the index if you delete the container.

```yaml
volumes:
- ./.rag_data/es_data:/usr/share/elasticsearch/data
```