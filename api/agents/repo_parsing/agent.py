import os
import json
from pathlib import Path
import requests # Keep for potential fallbacks
from bs4 import BeautifulSoup # Keep for potential fallbacks
from github import Github, GithubException # Import Github and its exception
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv
from .models import GitHubProfileData, GitHubRepo # Import the GitHub models

# --- GitHub API Setup ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Warning: GITHUB_TOKEN environment variable not set. API calls will be rate-limited or fail.")
    g = None
else:
    try:
        g = Github(GITHUB_TOKEN)
        # Test connection by getting authenticated user (optional)
        auth_user = g.get_user()
        print(f"GitHub API authenticated as: {auth_user.login}")
    except GithubException as e:
        print(f"Error initializing GitHub API client: {e}. Check your GITHUB_TOKEN.")
        g = None
    except Exception as e:
        print(f"An unexpected error occurred during GitHub client initialization: {e}")
        g = None
# --- ---

# --- LLM Setup (Reuse or define specific one) ---
# Ensure DEEPSEEK_API_KEY is set in your .env
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
     raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

# Reusing the model definition style from resume agent
# Ensure the API key is correctly passed
model = OpenAIModel(
    model_name='deepseek-chat',
    provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY)
)

github_agent = Agent(
    # model=model, # Use the pydantic-ai model instance if preferred
    model="gemini-2.5-flash", # Or your preferred model string
    deps_type=str, # Input to the agent run will be the prompt string
    system_prompt=(
        "You are an expert GitHub profile analyzer. Your task is to extract key information "
        "from structured data fetched via the GitHub API, supplemented potentially by profile README content. "
        "Focus on extracting profile details like name, bio, location, repo count, followers, following, and identify key skills or technologies "
        "mentioned in the bio or README. Also, list some notable repositories with their descriptions and languages. "
        "Return the response as a structured JSON object matching the provided schema."
    ),
)

# --- Fallback Scraping Function (Keep for robustness if API fails) ---
def fetch_github_profile_content_fallback(username: str) -> str:
    """Fetches the HTML content of a GitHub profile page. Basic scraping as fallback."""
    url = f"https://github.com/{username}"
    print(f"Attempting fallback scraping for {username}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Simplified extraction for fallback
        body_text = soup.body.get_text(separator='\n', strip=True)[:3000]
        return f"URL: {url}\n\n{body_text}"
    except requests.RequestException as e:
        print(f"Fallback scraping failed for {username}: {e}")
        return f"Error: Fallback scraping failed for {username}. URL: {url}"    
    
# --- ---

@github_agent.tool_plain
async def parse_github_profile(username: str) -> dict:
    """
    Parses a GitHub profile using the GitHub API (preferred) or basic scraping (fallback)
    and LLM analysis. Requires a GITHUB_TOKEN environment variable for API usage.
    """
    if not username:
        return {"error": "GitHub username is required."}

    profile_data_text = f"Analysis for GitHub user: {username}\n\n"
    api_used = False

    # --- Attempt to use GitHub API ---
    if g:
        try:
            print(f"Fetching GitHub data for '{username}' via API...")
            user = g.get_user(username)
            print(f"User: {user}")
            api_used = True

            profile_data_text += f"Name: {user.name}\n"
            profile_data_text += f"Username: {user.login}\n"
            profile_data_text += f"Bio: {user.bio}\n"
            profile_data_text += f"Location: {user.location}\n"
            profile_data_text += f"Public Repos: {user.public_repos}\n"
            profile_data_text += f"Followers: {user.followers}\n"
            profile_data_text += f"Following: {user.following}\n"
            profile_data_text += f"Company: {user.company}\n"
            profile_data_text += f"Blog/Website: {user.blog}\n"
            profile_data_text += f"Twitter: {user.twitter_username}\n"
            profile_data_text += f"Created At: {user.created_at}\n"
            profile_data_text += f"Updated At: {user.updated_at}\n"

            # Fetch top repositories (e.g., sorted by stars or updated date)
            profile_data_text += "\nTop Repositories (by stars):\n"
            try:
                # Sort by stars, descending. Limit to avoid excessive data.
                repos = user.get_repos(sort="stars", direction="desc")
                repo_count = 0
                for repo in repos:
                    if repo_count >= 10: # Limit number of repos in context
                        break
                    profile_data_text += (
                        f"- Name: {repo.name}\n"
                        f"  URL: {repo.html_url}\n"
                        f"  Description: {repo.description}\n"
                        f"  Language: {repo.language}\n"
                        f"  Stars: {repo.stargazers_count}\n"
                        f"  Forks: {repo.forks_count}\n"
                        f"  Created: {repo.created_at}\n"
                        f"  Updated: {repo.updated_at}\n"
                    )
                    repo_count += 1
            except GithubException as repo_e:
                 print(f"Could not fetch repositories for {username}: {repo_e}")
                 profile_data_text += "(Could not fetch repositories)\n"


            # Fetch profile README content if exists
            try:
                readmeRepo = user.get_repo(f"{username}/{username}") # Assuming the README is in the user's repo
                readme = readmeRepo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8')
                profile_data_text += f"\nProfile README Content (partial):\n---\n{readme_content[:1500]}\n---\n" # Limit README size
            except GithubException as readme_e:
                if readme_e.status == 404:
                    profile_data_text += "\n(No public profile README found)\n"
                else:
                    print(f"Error fetching profile README for {username}: {readme_e}")
                    profile_data_text += "\n(Error fetching profile README)\n"
            except Exception as readme_e: # Catch other potential decoding errors
                 print(f"Error processing profile README for {username}: {readme_e}")
                 profile_data_text += "\n(Error processing profile README)\n"

        except GithubException as e:
            print(f"GitHub API error for {username}: {e}. Status: {e.status}")
            if e.status == 404:
                 return {"error": f"GitHub user '{username}' not found."}
            # Fallback to scraping if API fails for reasons other than 404
            print("Falling back to scraping due to API error.")
            profile_data_text = fetch_github_profile_content_fallback(username)
            api_used = False
        except Exception as e:
             # Catch unexpected errors during API interaction
             print(f"Unexpected error during GitHub API fetch for {username}: {e}")
             print("Falling back to scraping.")
             profile_data_text = fetch_github_profile_content_fallback(username)
             api_used = False
    else:
        # --- Use Basic Scraping if API client not available ---
        print("GitHub API client not available. Using fallback scraping.")
        profile_data_text = fetch_github_profile_content_fallback(username)
        api_used = False

    # Check if fetching/scraping resulted in an error message
    if profile_data_text.startswith("Error:"):
        return {"error": profile_data_text}

    # Limit overall context size before sending to LLM
    profile_content_for_llm = profile_data_text[:4000]

    source_info = "GitHub API" if api_used else "Web Scraping (Fallback)"
    prompt = f"""
    Analyze the following data for GitHub user '{username}', obtained via {source_info}.
    Extract structured information based on this data.

    Profile Data:
    ---
    {profile_content_for_llm}
    ---

    Return ONLY a valid JSON object matching this schema. Do NOT include explanations, comments, or markdown formatting.
    Infer skills/tags primarily from the bio and profile README content if available.
    If information is missing or not applicable based on the provided data, use null or empty lists.

    Schema:
    {{
        "username": "{username}",
        "name": "Full Name or null",
        "bio": "User bio or null",
        "location": "User location or null",
        "public_repos_count": integer_or_null, // Extract from API data if present
        "followers": integer_or_null, // Extract from API data if present
        "following": integer_or_null, // Extract from API data if present
        "top_repositories": [ // List based on API data if present, otherwise empty
            {{
                "name": "Repo Name",
                "description": "Repo description or null",
                "url": "Full repo URL",
                "language": "Primary language or null",
                "stars": integer_or_null
            }}
        ],
        "skills_tags": ["tag1", "tag2", ...] // Inferred from bio/README
    }}
    """

    try:
        # print(f"Sending prompt to LLM for GitHub user: {username}")
        agent_result = await github_agent.run(prompt)

        if not isinstance(agent_result.data, str):
            print(f"Error: GitHub LLM returned {type(agent_result.data)} - {agent_result.data}")
            return {"error": f"GitHub LLM did not return a string, got type {type(agent_result.data)}"}

        raw_json = agent_result.data
        # Clean potential markdown fences
        cleaned_json = raw_json.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[len("```json"):].strip()
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-len("```")].strip()

        # Add extra print for debugging JSON issues
        # print("--- Cleaned GitHub LLM JSON START ---")
        # print(cleaned_json)
        # print("--- Cleaned GitHub LLM JSON END ---")

        parsed_data = json.loads(cleaned_json)

        # Ensure username matches (sometimes LLMs might hallucinate)
        parsed_data["username"] = username

        # Validate with Pydantic model
        github_data = GitHubProfileData(**parsed_data)
        print(f"Successfully parsed GitHub data for {username}")
        return github_data.model_dump()

    except json.JSONDecodeError as e:
        print(f"GitHub JSON Decode Error for {username}: {e}")
        print(f"Invalid JSON string received (after cleaning): {cleaned_json}")
        return {"error": "Error parsing GitHub LLM JSON response"}
    except Exception as e: # Catch Pydantic validation errors and others
        import traceback
        print(f"Error processing GitHub data for {username}: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        return {"error": f"Error processing GitHub data: {str(e)}"}