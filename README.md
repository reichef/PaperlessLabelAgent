# PaperlessLabelAgent

The paperless label agent classifies documents for [Paperless-ngx](https://docs.paperless-ngx.com/), proposing tags, a correspondent and a document type for each file in a dedicated folder.
It asks you to confirm every proposition before finalizing the classification.

It reads the tags, correspondents and document types that already exist in your Paperless-ngx instance.
Then, the Agent uses a local LLM (via [Ollama](https://ollama.com/)) to either match a document against those existing entities or, if no enity fits, propose new ones. You review every proposal for its correctness.
Any rejected entity (either existing or new one) is reclassified automatically, with your rejection fed back into the next attempt so the model doesn't repeat the same mistake.
Repetion is constrained by a threshold value. 

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/), running locally with a model pulled that supports structured/tool output
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed locally
- A running [Paperless-ngx](https://docs.paperless-ngx.com/) instance — or use mock mode (see below) to try the agent without one

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate      # on Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e .
```

## Configuration

Create a `.env` file in the project root with the following keys:
| Key | Description |
| -------- | -------- |
| `API_URL` | Base URL of your Paperless-ngx API, e.g. `http://localhost:8000/api` |
| `ACCOUNT` | Paperless-ngx username |
| `PASSWORD` | Paperless-ngx password |
| `MODEL` | Ollama model tag to use for classification, e.g. `qwen3.5:9b` or `llama3:8b` |
| `TESSDATA_PATH` | Path to your Tesseract `tessdata` directory |
| `INPUT_FOLDER` | Folder containing the PDF documents to classify |

**Mock mode**: if `ACCOUNT` and `PASSWORD` both contain the string `mock`, the agent skips the real Paperless-ngx API entirely and loads sample tags/correspondents/document types from `test/paperless-instance-mock` instead. 
The corresponding files are named `paperless_entity_mock_[correspondents | documenttypes | tags]` respectively. 
These files must contain valid json content according to the Paperless-ngx API.
This functionality is used to omit the need for a running paperless-ngx instance. 


## Usage

```bash
python -m paperlesslabelagent.agent
```

The agent will fetch your existing entities, process every PDF in `INPUT_FOLDER`, and then walk promts each proposal in the terminal, asking `[y/n]` questions as needed. Once every file is either confirmed or has exhausted its retry attempts, the run ends with a summary of what was and wasn't resolved.

## Current limitations

- Only PDF files are currently supported.
- No functionality yet to push anything back to Paperless-ngx (no document upload or entity-creation) (open TODO). 

## License

[Eclipse Public License 2.0](LICENSE)
