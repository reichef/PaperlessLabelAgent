

import os
from typing import Any

from dotenv import load_dotenv
from ollama import chat
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage
from langchain.tools import tool
import pymupdf
from tools.paperlesstools import list_tags_correspondents_and_document_types


load_dotenv()

url = os.getenv("API_URL")
account = os.getenv("ACCOUNT")
password = os.getenv("PASSWORD")
input_folder = os.getenv("INPUT_FOLDER")

MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = f"""You are an agent that labels documents in Paperless-ngx.
Rules:
- Read the labels, correspondents and document types from Paperless-ngx from the API at {url} with the account {account} and the password {password}.
- Analyze the files in the folder {input_folder} and determine which labels, correspondents and document types fit the files best. If no matching label, correspondent or document type is found, output an appropriate message for the corresponding file.
"""



@tool
def translate_files_in_input_folder(input_folder: str) -> dict[str, Any]:
    """Analyzes the files in the input folder and returns for each file the contained text."""

    if not os.path.exists(input_folder):
        raise RuntimeError(f"Input folder '{input_folder}' does not exist.")

    fileContents = {}
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        #TODO We can add other file types supported by Paperless-ngx here, but for now, we will just handle PDF files.
        content = extractTextFromPDF(filename, file_path)
        fileContents[filename] = content

    return fileContents

def extractTextFromPDF(filename: str, file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    if not os.path.isfile(file_path):
        raise RuntimeError(f"File '{file_path}' does not exist.")

    if not filename.lower().endswith((".pdf")):
        raise RuntimeError(f"File '{file_path}' is not a PDF file.")

    document = pymupdf.open(file_path)
    text = ""

    for page in document:
        text += page.get_text()
    # TODO We can also add some logic here to analyze the images in the PDF, but for now, we will just extract the text.

    return text

def run_agent() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    response = chat(model=MODEL, messages=messages, tools=[list_tags_correspondents_and_document_types], think=True)

    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        result = list_tags_correspondents_and_document_types(**call.function.arguments)
        messages.append(response.message)
        messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

        final_response = chat(
            model=MODEL,
            messages=messages,
            tools=[list_tags_correspondents_and_document_types],
            think=True,
        )
        print(final_response.message.content)
        return

    print(response.message.content or "No answer received from the model.")


if __name__ == "__main__":
    run_agent()

