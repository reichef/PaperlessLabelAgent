import os
from dotenv import load_dotenv
from paperlesslabelagent.core.driver import run_interactive
from paperlesslabelagent.strategies.iterative.graph import graph as iterative_graph


load_dotenv(override=True)


input_folder = os.getenv("INPUT_FOLDER")

if __name__ == "__main__":
    result = run_interactive(iterative_graph, {"input_folder": f"{input_folder}"}, thread_id="paperless-label-run")

    unconfirmed = [filename for filename, proposal in result.get("proposals", {}).items() if not proposal.get("confirmed")]
    if unconfirmed:
        print(f"\nDone — {len(unconfirmed)} file(s) hit the retry limit and remain unconfirmed: {', '.join(unconfirmed)}")
    else:
        print("\nDone — all proposals confirmed.")
