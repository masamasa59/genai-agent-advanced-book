# Practical Introduction to AI Agents for Field Use - Chapter 7

This directory contains source code and resources related to Chapter 7 of the book "Practical Introduction to AI Agents for Field Use" (Kodansha).

In Chapter 7, you will learn how to build multi-agent systems through the implementation of a **decision support agent** and a **personalization policy support agent**.

To run the code in Chapter 7, follow the steps below.

## Prerequisites

To run this project, you will need the following:

- Python 3.12 or higher
- VS Code
- Opened as a workspace using VS Code's Multi-root Workspaces feature (see [here](../README.md) for instructions)
- OpenAI account and API key

Python dependencies are listed in `pyproject.toml`.

## Environment Setup

### 1. Open the chapter 7 workspace
Create a virtual environment in the chapter7 directory.
Select `chapter7` in the Add Terminal in VS Code.

### 2. Installing uv

We use `uv` to resolve dependencies.
If you have never used `uv` before, install it as follows.

Using `pip`:
```bash
pip install uv
```

Mac or Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Creating a Python Virtual Environment and Installing Dependencies

Installing Dependencies
```bash
uv sync
```

Activate the virtual environment created after installation.

``bash
source .venv/bin/activate
```

### 4. Setting Environment Variables
Copy the .env.example file and create a `.env` file with the following content:

If you do not have an OpenAI API key, obtain one from the [OpenAI official website](https://platform.openai.com/).

```bash

```env
# OpenAI API settings
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_MODEL= "gpt-4o-2024-08-06"
```

## Directory structure

```
chapter7/
├── .env.sample # Environment variable sample file
├── .python-version # Python version specification
├── pyproject.toml # Python project settings
├── uv.lock # Dependency lock file
├── notebooks/ # Jupyter Notebook files
│ ├── decision_support_agent_runner.ipynb # Decision support agent execution example
│ └── macrs_runner.ipynb # MACRS execution example
└── src/ # Source code
├── custom_logger.py # Custom Logger
├── decision_support_agent/ # Decision Support Agent
│ ├── agent.py # Agent Main Logic
│ ├── configs.py # Configuration File
│ ├── models.py # Data Model
│ └── prompts.py # Prompt Template
└── macrs/ # MACRS (Multi-Agent Collaborative Review System)
├── agent.py # Personalized Policy Support Agent Main Logic
├── configs.py # Configuration File
├── custom_logger.py # Custom Logger
├── models.py # Data Model
└── prompts.py # Prompt Template
```

## Usage

### Decision Support Agent

1. Open `notebooks/decision_support_agent_runner.ipynb`.
2. Run the cells sequentially to verify the behavior of the decision support agent.

### MACRS (Multi-Agent Collaborative Review System)

1. Open `notebooks/macrs_runner.ipynb`.
2. Run the cells sequentially to verify the behavior of the personalization support agent.

## Main Features

### Decision Support Agent
- Persona-based, multifaceted opinion generation
- Automatic generation of improvement proposals
- Complex workflow management using LangGraph

### MACRS
- Multi-agent collaborative review system
- Efficient review process with asynchronous processing

## Notes

- An OpenAI API key is required. Please obtain it in advance.
- Please work with a Python virtual environment enabled.
- For details, please refer to the relevant chapter in the book.