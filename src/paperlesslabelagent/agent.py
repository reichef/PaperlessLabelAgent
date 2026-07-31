import os
from dotenv import load_dotenv
from paperlesslabelagent.graph import graph


load_dotenv()


input_folder = os.getenv("INPUT_FOLDER")

if __name__ == "__main__":
    graph.invoke({"input_folder" : f"{input_folder}"})

