
import json
import os
from typing import Any

import requests
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

url = os.getenv("API_URL")
account = os.getenv("ACCOUNT")
password = os.getenv("PASSWORD")

MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = """Du bist ein Agent, der mir die Informationen über Labels, Correspondents und Document Types in Paperless-ngx liefert.
Regeln:
- Lese die Labels, Correspondents und Document Types aus Paperless-ngx aus. Benutze hierfür das Tool list_tags_correspondents_and_document_types.
- Gib diese Informationen in deiner Antwort an den Benutzer zurueck. Fasse die Ergebnisse kurz und verständlich zusammen.
"""


def list_tags_correspondents_and_document_types() -> dict[str, Any]:
    """Liefert Labels, Correspondents und Document Types aus Paperless-ngx."""
    if not url:
        raise RuntimeError("API_URL ist nicht gesetzt. Bitte .env prüfen.")

    label_url = f"{url}/tags/"
    correspondent_url = f"{url}/correspondents/"
    document_type_url = f"{url}/document_types/"

    response_labels = requests.get(label_url, auth=(account, password), timeout=10)
    response_correspondents = requests.get(correspondent_url, auth=(account, password), timeout=10)
    response_document_types = requests.get(document_type_url, auth=(account, password), timeout=10)

    response_labels.raise_for_status()
    response_correspondents.raise_for_status()
    response_document_types.raise_for_status()

    return {
        "labels": response_labels.json(),
        "correspondents": response_correspondents.json(),
        "document_types": response_document_types.json(),
    }


def build_summary_text(data: dict[str, Any]) -> str:
    """Erstellt eine einfache, lesbare Zusammenfassung der Paperless-Daten."""
    labels = data.get("labels", {}).get("results", [])
    correspondents = data.get("correspondents", {}).get("results", [])
    document_types = data.get("document_types", {}).get("results", [])

    parts: list[str] = []
    for item in labels:
        parts.append(f"Label: {item.get('name', '')}")
    for item in correspondents:
        parts.append(f"Correspondent: {item.get('name', '')}")
    for item in document_types:
        parts.append(f"Document Type: {item.get('name', '')}")

    return "\n".join(parts)


def run_agent() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    response = chat(model=MODEL, messages=messages, tools=[list_tags_correspondents_and_document_types], think=True)

    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        result = list_tags_correspondents_and_document_types()
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

    print(response.message.content or "Keine Antwort vom Modell erhalten.")


if __name__ == "__main__":
    run_agent()

