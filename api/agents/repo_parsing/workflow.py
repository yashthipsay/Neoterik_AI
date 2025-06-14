from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict
from .agent import parse_github_profile
from .models import GitHubProfileData

# Define the state specifically for GitHub parsing
class GitHubParsingState(TypedDict):
    github_username: Annotated[str | None, "GitHub username to parse"]
    parsed_github_data: Annotated[GitHubProfileData | Dict | None, "Parsed data from GitHub profile or error dict"]
    error: Annotated[str | None, "Error message if parsing fails"]

# Define the node for GitHub parsing
async def parse_github_node(state: GitHubParsingState):
    """Parses the GitHub profile using the GitHub agent tool."""
    username = state.get("github_username")
    if not username:
        return {"error": "GitHub username not provided in state."}
    try:
        print(f"Workflow: Starting GitHub parsing for {username}")
        github_data = await parse_github_profile(username)
        # Check if the agent returned an error dictionary
        if isinstance(github_data, dict) and 'error' in github_data:
             print(f"Workflow: GitHub parsing agent returned error: {github_data['error']}")
             return {"error": f"GitHub Parsing Failed: {github_data['error']}", "parsed_github_data": None}
        print(f"Workflow: Successfully parsed GitHub data for {username}")
        return {"parsed_github_data": github_data, "error": None}
    except Exception as e:
        import traceback
        print(f"Workflow: Error during GitHub parsing node execution for {username}: {e}")
        print(traceback.format_exc())
        return {"error": f"GitHub Parsing Node Failed: {str(e)}", "parsed_github_data": None}

# Create the workflow graph
def create_github_parsing_workflow():
    """Creates the LangGraph workflow for parsing GitHub profiles."""
    workflow = StateGraph(GitHubParsingState)

    # Add the single node for parsing
    workflow.add_node("parse_github", parse_github_node)

    # Set the entry point
    workflow.set_entry_point("parse_github")

    # Add an edge from the parsing node to the end
    # Since it's a single-step process for now
    workflow.add_edge("parse_github", END)

    # Compile the graph
    print("Compiling GitHub parsing workflow...")
    app = workflow.compile()
    print("GitHub parsing workflow compiled.")
    return app