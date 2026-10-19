from paperlesslabelagent.core.nodes import entities as entities_module
from paperlesslabelagent.core.nodes.entities import load_documents


def test_load_documents_skips_processed_subfolder(tmp_path, monkeypatch):
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 invoice")
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "earlier.pdf").write_bytes(b"%PDF-1.4 earlier")

    extracted_filenames = []

    def fake_extract_text_from_pdf(filename, file_path):
        extracted_filenames.append(filename)
        return f"text of {filename}"

    monkeypatch.setattr(entities_module, "extractTextFromPDF", fake_extract_text_from_pdf)

    result = load_documents({"input_folder": str(tmp_path)})

    assert result == {"file_texts": {"invoice.pdf": "text of invoice.pdf"}}
    assert extracted_filenames == ["invoice.pdf"]
