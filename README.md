# "A Practical Introduction to AI Agents for Field Use" Repository

<img src=".github/assets/cover.jpg" alt="Book Cover" width="300">

This is the repository for ["A Practical Introduction to AI Agents for Field Use" (Kodansha)] (www.amazon.co.jp/dp/4065401402).

This repository provides code and configuration files for each chapter of the book, focusing on the implementation chapters.

## How to Use This Repository

This repository is intended for use with VSCode.

This repository utilizes VSCode's multi-root workspaces (hereafter referred to as workspaces) to manage each chapter independently.

Working in a workspace can be done in one of the following ways:
- Launch VS Code by double-clicking the `genai-book.code-workspace` file in this repository in Finder or Explorer.
- Select `File > Open Workspace with File` from the VS Code menu and select the `genai-book.code-workspace` file.

The code for each chapter is stored in the `chapter<chapter number>` directory and can be opened individually as a workspace.

To run the code, refer to the README.md for each chapter to confirm environment setup and execution instructions. Please note in particular that the Python virtual environment is intended to be created within each chapter's workspace.

## Support
- Questions and bug reports are accepted through GitHub Issues. If you have any questions, please see [issues](https://github.com/masamasa59/genai-agent-advanced-book/issues).
- Typo listings in the book (which have been corrected) are listed in the [Typo List (Closed)](https://github.com/masamasa59/genai-agent-advanced-book/issues?q=state%3Aclosed%20label%3A%E8%AA%A4%E6%A4%8D). Typo listings before corrections are listed in the [Typo List (Open)](https://github.com/masamasa59/genai-agent-advanced-book/issues?q=state%3Aopen%20label%3A%E8%AA%A4%E6%A4%8D). Please refer to these as well.
