# 🤖 NL2Bash — Natural Language to Terminal Commands

NL2Bash is an agentic AI assistant that translates natural language queries into accurate bash commands using semantic similarity and machine learning. Ideal for beginners, power users, and anyone who wants to streamline their command-line workflow.

## 🚀 Features
- Translate human-like queries to shell commands (e.g., "list all files" → `ls`)
- Uses `SentenceTransformer` embeddings for semantic understanding
- Ranks and displays top command matches with similarity scores
- Rich, interactive terminal UI

## 🧠 How It Works
1. User enters a natural language description of a task.
2. The system embeds the input using a transformer model.
3. It compares against a dataset of known queries and bash commands.
4. Top-k most similar commands are suggested in a ranked table.

## 📦 Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
