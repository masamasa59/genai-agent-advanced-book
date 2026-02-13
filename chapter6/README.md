# Practical Introduction to AI Agents for Field Use - Chapter 6

This directory contains source code and resources related to Chapter 6 of the book "Practical Introduction to AI Agents for Field Use" (Kodansha).

## Prerequisites

To run this project, you will need the following:

- Python 3.11 or higher
- OpenAI API key
- Anthropic API key
- Cohere API key
- Jina Reader API key
- LangSmith API key (optional)
- VS Code

Python dependencies are listed in `pyproject.toml`.

## Open in VS Code Workspace

Launch VS Code and open the project root directory (`genai-agent-advanced-book`) as a workspace. Then, add a terminal and select `chapter6` to efficiently edit and run the code in this chapter.

## Environment Setup

### Creating a Python Virtual Environment and Installing Dependencies

`uv` is used to resolve dependencies. To install `uv`, follow these steps:

**Using pip:**

```bash
pip install uv
```

**For Mac or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

```bash
uv sync
```

### Setting Environment Variables

Copy the `.env.sample` file to `.env` and set the necessary environment variables:

```
$ cp .env.sample .env
```

Set the following items in the `.env` file:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `COHERE_API_KEY`: Cohere API key
- `JINA_API_KEY`: Jina Reader API key

The following items are optional. Please set this when you want to check the integration between LangGraph Studio and LangSmith.

- `LANGSMITH_API_KEY`: LangSmith API key

## Directory structure

```
chapter6/
├── .env.sample # Environment variable sample file
├── pyproject.toml # Python project configuration
├── uv.lock # Dependency lock file
├── langgraph.json # LangGraph configuration file
├── README.md # This file
├── arxiv_researcher/ # arXiv Paper Researcher main package
│ ├── __init__.py
│ ├── agent/ # Agent implementation
│ │ ├── __init__.py
│ │ ├── paper_analyzer_agent.py # Paper analysis agent
│ │ ├── paper_search_agent.py # Paper search agent
│ │ └── research_agent.py # Research Agent (Main)
│ ├── chains/ # LangChain Chain Implementation
│ │ ├── goal_optimizer_chain.py # Goal Optimizer Chain
│ │ ├── hearing_chain.py # Hearing Chain
│ │ ├── paper_processor_chain.py # Paper Processing Chain
│ │ ├── prompts/ # Prompt Templates
│ │ │ ├── check_sufficiency.prompt
│ │ │ ├── goal_optimizer_conversation.prompt
│ │ │ ├── goal_optimizer_search.prompt
│ │ │ ├── hearing.prompt
│ │ │ ├── query_decomposer.prompt
│ │ │ ├── reporter_system.prompt
│ │ │ ├── reporter_user.prompt
│ │ │ ├── set_section.prompt
│ │ │ ├── summarize.prompt
│ │ │ └── task_evaluator.prompt
│ │ ├── query_decomposer_chain.py # Query decomposition chain
│ │ ├── reading_chains.py # Paper reading chain
│ │ ├── reporter_chain.py # Report generation chain
│ │ ├── task_evaluator_chain.py # Task evaluation chain
│ │ └── utils.py # Utility functions
│ ├── logger.py # Logger settings
│ ├── models/ # Data model
│ │ ├── __init__.py
│ │ ├── arxiv.py # arXiv related model
│ │ ├── markdown.py # Markdown related model
│ │ └── reading.py # Reading Relation Model
│ ├── searcher/ # Search Function
│ │ ├── arxiv_searcher.py # arXiv Search Implementation
│ │ └── searcher.py # Search Interface
│ ├── service/ # Service Layer
│ │ ├── markdown_parser.py # Markdown Parser
│ │ ├── markdown_storage.py # Markdown Storage
│ │ └── pdf_to_markdown.py # PDF to Markdown Conversion
│ └── settings.py # Settings File
├── fixtures/ # Test Data
│ ├── 2408.14317.md # Sample Paper (Markdown Format)
│ ├── sample_context.txt # Sample Context
│ └── sample_goal.txt # Sample Goal
├── logs/ # Log File Destination
│ └── application.log
└── storage/ # Storage
└── markdown/ # Paper Markdown File Destination
└── *.md # Processed Paper File
```

## Usage

### Running with LangGraph Studio

To run the agent using LangGraph Studio:

```
$ uv run langgraph dev --no-reload
```

Once LangGraph Studio is launched, access `http://localhost:8123` in your browser (it will open automatically). Enter your research goals in the browser UI, run the agent, and view the agent's progress in the browser.

### Main Features

#### arXiv Paper Researcher

This agent searches for relevant papers on arXiv, converts PDF files to Markdown format, and performs structured analysis. It uses a listening function to interactively clarify the user's research objectives, and then synthesizes the collected information to generate a report.

### Supplement: How to Test Subgraphs on the Command Line

You can test each subgraph individually to verify its operation.

#### 1. Testing PaperAnalyzerAgent (Paper Analysis Agent)

This test verifies the detailed analysis function for a single paper. This agent analyzes the paper's structure, generates a summary of each section, and extracts key points.

```
$ uv run python -m arxiv_researcher.agent.paper_analyzer_agent fixtures/2408.14317.md
```

When run, the agent displays the paper's title, authors, and abstract, and summarizes each section, such as the Introduction, Methodology, and Results. It also lists the main findings and contributions in bullet points, and outputs an analysis of the paper's strengths and limitations.

#### 2. Testing PaperSearchAgent (Paper Search Agent)

This test verifies the paper search function from arXiv. This agent retrieves relevant papers based on a search query and extracts basic metadata.

```
$ uv run python -m arxiv_researcher.agent.paper_search_agent
```

When executed, the agent searches for papers using a default search query (e.g., "LLM agent") and displays information about several papers. For each paper, you can view basic information such as the title, author, publication date, and arXiv ID, as well as the paper's abstract.

## Papers used in the test

The following papers were used in the test:

```bibtex
@article{dmonte2024claim,
title={Claim Verification in the Age of Large