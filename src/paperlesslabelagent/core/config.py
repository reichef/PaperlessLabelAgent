import os

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = os.getenv("MODEL")
TESSDATA_PATH = os.getenv("TESSDATA_PATH")
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES")
ENTITY_LANGUAGE = os.getenv("ENTITY_LANGUAGE")
