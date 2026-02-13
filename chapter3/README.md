# Introduction to Practical AI Agents, Chapter 3

This project is an environment for running the sample code from Chapter 3 of "Introduction to Practical AI Agents."

## Prerequisites

To run this project, you need the following:

- Python 3.12 or higher
- Docker and Docker Compose
- VSCode
- Opened as a workspace using VSCode's Multi-root Workspaces feature (see [here](../README.md) for instructions).
- OpenAI account and API key
- TAVILY account and API key

Python dependencies are listed in `pyproject.toml`.

## Environment Setup

### 1. Open the chapter 3 workspace
Create a virtual environment in the chapter3 directory.
Select `chapter3` in the Add Terminal in VSCode.

### 2. Install uv

`uv` is used to resolve dependencies.
If you have never used `uv` before, install it using the following method.

Using `pip`:
```bash
pip install uv
```

For Mac or Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Creating a Python Virtual Environment and Installing Dependencies

Installing Dependencies
```bash
uv sync
```

Activate the virtual environment created after installation.

```bash
source .venv/bin/activate
```

### 4. Setting Environment Variables
Copy the .env.example file and create a `.env` file with the following content:

```bash

```env
# OpenAI API Settings
# If you do not have an OpenAI API key, obtain one from the [OpenAI official website](https://platform.openai.com/).
OPENAI_API_KEY=your_openai_api_key

# Tavily API Settings (for web search)
# Create an account at https://tavily.com and obtain your API key.
TAVILY_API_KEY=your_tavily_api_key
```