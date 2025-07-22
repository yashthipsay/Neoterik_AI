import asyncio
from celery import Celery

from agents.langgraph_workflow.unified_workflow import build_graph

# Initalize the celery app
celery_app = Celery(
    "CoverLetterWorkflow",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://",
)

# Define the celery task

@celery_app.task(name="run_cover_letter_workflow")
def run_unified_workflow_task(initial_state: dict):
    """
    This Celery task takes the initial state and runs the entire
    LangGraph workflow asynchronously.
    """

    print("Starting the unified workflow task...")

    app = build_graph()

    final_state = asyncio.run(app.ainvoke(initial_state))

    print("Workflow finished. Returning final context.")
    # Return the 'context' dictionary from the final state, 
    # which contains the generated cover letter.
    return final_state.get("context", {})

