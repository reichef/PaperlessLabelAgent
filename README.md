# PaperlessLabelAgent

The paperless label agent classifies documents for [Paperless-ngx](https://docs.paperless-ngx.com/), proposing tags, a correspondent and a document type for each file in a dedicated folder.
It asks you to confirm every proposition before finalizing the classification.

It reads the tags, correspondents and document types that already exist in your Paperless-ngx instance.
Then, the Agent uses a local LLM (via [Ollama](https://ollama.com/)) to either match a document against those existing entities or, if no enity fits, propose new ones. You review every proposal for its correctness.
Any rejected entity (either existing or new one) is reclassified automatically, with your rejection fed back into the next attempt so the model doesn't repeat the same mistake.
Repetion is constrained by a threshold value.

The agent classifies, reviews and confirms one document fully before moving to the next (iterative strategy), so newly accepted entities are reused for later documents and every confirmed document is persisted immediately.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/), running locally with a model pulled that supports structured/tool output
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed locally
- A running [Paperless-ngx](https://docs.paperless-ngx.com/) instance — or use mock mode (see below) to try the agent without one

## Configuration

Create a `.env` file in the project root with the following keys:
| Key | Description |
| -------- | -------- |
| `API_URL` | Base URL of your Paperless-ngx API, e.g. `http://localhost:8000/api` |
| `ACCOUNT` | Paperless-ngx username |
| `PASSWORD` | Paperless-ngx password |
| `MODEL` | Ollama model tag to use for classification, e.g. `qwen3.5:9b` or `llama3:8b` |
| `TESSDATA_PATH` | Path to your Tesseract `tessdata` directory |
| `OCR_LANGUAGES` | Languages that should be handled by Tesseract OCR | 
| `INPUT_FOLDER` | Folder containing the PDF documents to classify |
| `ENTITY_LANGUAGE` | Language(s) the LLMs are promted to provide entities for, provided in comma seperated full text values (e.g., english, german) |
| `DELETE_INPUT_FILES_AFTER_UPLOAD` | Set to `true` to delete a file from `INPUT_FOLDER` once it has been uploaded to Paperless-ngx. Otherwise (default: `false`) the file is moved to `INPUT_FOLDER/processed/`; if a file of that name is already there, a numeric suffix is added (e.g. `invoice (1).pdf`) |


**Mock mode**: if `ACCOUNT` and `PASSWORD` both contain the string `mock`, the agent fetches sample tags/correspondents/document types from `test/paperless-instance-mock` instead of using the Paperless-ngx API on an running instance. 
The corresponding files are named `paperless_entity_mock_[correspondents | documenttypes | tags]` respectively. 
These files must contain valid json content according to the Paperless-ngx API.
This functionality is used to omit the need for a running paperless-ngx instance. 


## Usage

```bash
python -m paperlesslabelagent.agent
```

The agent will fetch your existing entities, process every PDF in `INPUT_FOLDER`, and then walk promts each proposal in the terminal, asking `[y/n]` questions as needed. Once every file is either confirmed or has exhausted its retry attempts, the run ends with a summary of what was and wasn't resolved.

Each document is uploaded to Paperless-ngx as soon as you confirm it, and then deleted or moved to `INPUT_FOLDER/processed/` (see `DELETE_INPUT_FILES_AFTER_UPLOAD`). If the agent is aborted, everything confirmed so far is already in Paperless-ngx and out of `INPUT_FOLDER`; start the agent again to continue with the remaining files. A confirmed document cannot be undone from within the agent — correct it directly in Paperless-ngx.

## Current limitations

- Only PDF files are supported.
  
## License

[Eclipse Public License 2.0](LICENSE)
