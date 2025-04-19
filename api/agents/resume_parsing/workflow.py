from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from .agent import parse_resume_from_pdf, validate_parsed_data # Assuming validate_parsed_data is sync for now
from .models import ResumeData

# Define the state for the graph
class ResumeParsingState(TypedDict):
    resume_path: Annotated[str | None, "Path to the resume file or None"]
    parsed_data: Annotated[dict | None, "Parsed data extracted from the resume"]
    validation_result: Annotated[dict | None, "Results from the validation step"]
    error: Annotated[str | None, "Error message if any step fails"]

# Define the nodes
async def parse_node(state: ResumeParsingState): # Make the node async
    """Parses the resume using the LLM agent tool."""
    resume_path = state.get("resume_path")
    try:
        # Await the async tool function
        parsed_data = await parse_resume_from_pdf(resume_path)
        return {"parsed_data": parsed_data, "error": None}
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}

def validate_node(state: ResumeParsingState):
    """Validates the parsed data using the validation tool."""
    parsed_data = state.get("parsed_data")
    if not parsed_data:
        return {"error": "No parsed data to validate."}
    try:
        # Assuming validate_parsed_data is synchronous
        # If it were async, you'd need to make this node async and await it
        validation_result = validate_parsed_data(parsed_data=parsed_data)
        return {"validation_result": validation_result, "error": None}
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

# Define conditional edges (if needed, otherwise simple sequence)
def should_validate(state: ResumeParsingState):
    """Decide whether to proceed to validation."""
    if state.get("error"):
        return "error_end" # Go directly to end if parsing failed
    return "validate" # Proceed to validation

# Create the workflow graph
def create_resume_parsing_workflow():
    workflow = StateGraph(ResumeParsingState)

    # Add nodes
    workflow.add_node("parse", parse_node)
    workflow.add_node("validate", validate_node)
    # workflow.add_node("error_end", END) # REMOVE THIS LINE

    # Set entry point
    workflow.set_entry_point("parse")

    # Add edges
    workflow.add_conditional_edges(
        "parse",
        should_validate,
        {
            "validate": "validate",
            "error_end": END, # Use END here to terminate the graph
        }
    )
    workflow.add_edge("validate", END) # Use END here to terminate the graph

    # Compile the graph
    app = workflow.compile()
    return app