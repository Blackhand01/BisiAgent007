# BisiAgent007

**RAG-based AI coding assistant — your intelligent pair programming companion.**

## Overview

BisiAgent007 is a minimal yet powerful chat agent leveraging OpenAI GPT-4o and a suite of Retrieval-Augmented Generation (RAG) tools (`rag_tools.py`). Seamlessly integrated into a VS Code–like environment, it automates semantic code search, file operations, and context-aware code modifications.

## Key Features

* **GPT-4o Integration**: Uses the `gpt-4o-mini` model for natural language responses and autonomous tool invocation.
* **Comprehensive RAG Tools**: Offers functions for semantic search, file reading, code editing, grep-based search, shell command execution, and more.
* **Automatic Schema Generation**: Inspects `rag_tools.py` function signatures to produce precise JSON schemas for tool calls.
* **Persistent Conversation History**: Tracks all interactions (`system`, `user`, `assistant`, `tool`) to maintain context and continuity.

## Requirements

* **Python**: Version 3.8 or higher
* **Ripgrep**: `rg` command-line tool for text-based fallback searches
* **Dependencies** (install via `requirements.txt`):

  * `openai`
  * `python-dotenv`
  * `numpy`, `scikit-learn` (for semantic vector search)
  * Standard libraries: `textwrap`, `logging`, `difflib`, etc.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/BisiAgent007.git
   cd BisiAgent007
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   * Create a `.env` file in the project root:

     ```dotenv
     OPENAI_API_KEY=sk-...
     PROJECT_ID=proj_...
     ORGANIZATION_ID=org_...
     RAG_LOG_LEVEL=INFO
     RAG_ALLOW_AUTO_CMDS=1
     ```

## Project Structure

```
├── rag_tools.py          # RAG tool implementations
├── system_prompt.txt     # System-level prompt customization
├── chat_agent.py         # Entry point for the chat interface
├── requirements.txt      # Dependency list
└── README.md             # Project documentation
```

## Usage

1. **Adjust working directory** in `chat_agent.py`:

   ```python
   os.chdir("/path/to/your/codebase")
   ```

2. **Launch the agent**:

   ```bash
   python chat_agent.py
   ```

3. **Interact naturally**:

   * Enter your programming questions or commands.
   * The assistant will reply or trigger appropriate RAG tool calls in JSON.

## Advanced Configuration

* **System Prompt**: Tailor `system_prompt.txt` to modify agent behavior.
* **Logging**: Use `RAG_LOG_LEVEL=DEBUG` for verbose output.
* **Semantic Search**: Ensure `OPENAI_API_KEY` and ML dependencies (`numpy`, `scikit-learn`) are installed.

## Example Interaction

```text
👤 Find all occurrences of `compute_embedding` in Python files.
🤖 Executing `grep_search`...
🤖 Results:
    • src/utils/embeddings.py:42: def compute_embedding(text):
    • src/helpers/rag.py:17: def compute_embedding_chunk(chunk):
```

## Contributing

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit changes: `git commit -m "Add my feature"`.
4. Push to your fork: `git push origin feature/my-feature`.
5. Open a Pull Request.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

*Documentation prepared by Stefano Roy Bisignano*
