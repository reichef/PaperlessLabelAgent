import os
from dotenv import load_dotenv
from langgraph.types import Command
from paperlesslabelagent.graph import graph
from paperlesslabelagent.nodes.review import collect_hallucination_answer, collect_review_answer


load_dotenv()


input_folder = os.getenv("INPUT_FOLDER")
config = {"configurable": {"thread_id": "paperless-label-run"}}

if __name__ == "__main__":
    result = graph.invoke({"input_folder": f"{input_folder}"}, config=config)

    while "__interrupt__" in result:
        interrupt_content = result["__interrupt__"][0].value
        if interrupt_content["kind"] == "hallucination":
            answer = collect_hallucination_answer(interrupt_content)
        else:  # "verify"
            answer = collect_review_answer(interrupt_content)
        result = graph.invoke(Command(resume=answer), config=config)

    unconfirmed = [filename for filename, proposal in result.get("proposals", {}).items() if not proposal.get("confirmed")]
    if unconfirmed:
        print(f"\nDone — {len(unconfirmed)} file(s) hit the retry limit and remain unconfirmed: {', '.join(unconfirmed)}")
    else:
        print("\nDone — all proposals confirmed.")

