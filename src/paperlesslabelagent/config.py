import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL")
TESSDATA_PATH = os.getenv("TESSDATA_PATH")
OCR_LANGUAGES = "deu+eng+fra"
