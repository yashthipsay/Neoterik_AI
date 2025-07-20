import os, re, json, textwrap, asyncio
from typing import List, TypedDict
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
# from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 API Keys
# os.environ["TAVILY_API_KEY"] = "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"
# os.environ["GROQ_API_KEY"] = ""

class JobPageAnalysis(BaseModel):
    """
    A schema to analyze a webpage and determine if it's a specific job posting.
    This AI-powered check is more reliable than keyword matching.
    """
    is_job_posting: bool = Field(description="True if the page is a specific job posting, False otherwise.")
    confidence_score: float = Field(description="A 0.0-1.0 confidence score on whether this is a job page.")
    detected_job_title: str = Field(description="The job title detected on the page, if any.")
    detected_company_name: str = Field(description="The hiring company name, avoiding job boards like LinkedIn.")
    reasoning: str = Field(description="A brief explanation for the is_job_posting decision.")

# 📤 Output Schema
class CompanyResearchOutput(BaseModel):
    company_name: str
    job_title: str
    job_description: str
    company_summary: str
    preferred_qualifications: List[str] = Field(default_factory=list)
    skillset: List[str] = Field(default_factory=list)
    company_vision: str = ""
    additional_notes: str = ""

class GraphState(TypedDict):
    job_url: str
    is_job_page: bool
    scraped_html_text: str
    job_title: str
    company_name: str
    search_results: str
    job_search_results: str
    linkedin_results: str
    messages: List
    final_output: CompanyResearchOutput

# 🧼 Utilities
def clean_text(html: str) -> str:
    """
    Cleans HTML content by removing script/style tags and excess whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)

def parse_llm_output(output: str) -> CompanyResearchOutput:
    def extract(label):
        match = re.search(rf"\*\*{label}:\*\*\s*(.*?)(?=\n\*\*|\Z)", output, re.DOTALL)
        return clean_text(match.group(1)) if match else ""

    def extract_list(label):
        match = re.search(rf"\*\*{label}:\*\*\s*((?:\n- .+)+)", output)
        return [clean_text(line[2:]) for line in match.group(1).splitlines() if line.strip()] if match else []

    return CompanyResearchOutput(
        company_name=extract("Company Name"),
        job_title=extract("Job Title"),
        job_description=extract("Job Description"),
        company_summary=extract("Company Summary"),
        preferred_qualifications=extract_list("Preferred Qualifications"),
        skillset=extract_list("Skillset"),
        company_vision=extract("Company Vision"),
        additional_notes=extract("Additional Notes")
    )

# 🤖 Agent Setup
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.1)
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key="AIzaSyANJ3NXIcSwHrcXMjFGueLNEEJh_pgFz70")
search_tool = TavilySearchResults(max_results=5)

react_agent = create_react_agent(llm, tools=[search_tool], prompt="""
You are a job and company research expert. Extract real info and format:
Strictly Follow RULES:
1. Extract only REAL information - no placeholders.
2. If any field is missing or incomplete, use the search tool to retrieve that information.
3. Avoid using "not mentioned" or "not found", "not explicitly provided" — instead, intelligently find from other sources.
4. Follow this exact format: And You should Return all the information in following format only, dont miss any fields!! And dont give Half information give full information you should complete with addtional notes at the end.
**Company Name:** <actual>
**Job Title:** <actual>
**Job Description:**
  <description text>
**Company Summary:**
  <company description>
**Preferred Qualifications:** Yrs of Experience mentioned, Educational Qualifications if any mentioned.
- <qualification 1>
- <qualification 2>
**Skillset:**
- <skill 1>
- <skill 2>
**Company Vision:**
  <vision or mission>
**Additional Notes:**
  <perks, hybrid policy, DEI, etc.>
""")

# 🌐 Scraper
async def scrape_with_playwright(url: str) -> str:
    """
    Asynchronously scrapes a URL using Playwright to handle dynamic, JS-heavy pages.
    """
    print(f" Scraping URL: {url}...")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=90000, wait_until='networkidle')
            html = await page.content()
            await browser.close()
            print(" Scrape successful.")
            return html
    except Exception as e:
        print(f" Scraping error: {e}")
        return ""

def extract_company_name(soup: BeautifulSoup, job_title: str, body_text: str) -> str:
    title_match = re.search(r'[@|at]\s+([A-Za-z0-9&\-\.\'\s]+)', job_title, re.IGNORECASE)
    if title_match:
        print("extract company done by method 1")
        return title_match.group(1).strip()

    body_match = re.search(r"(?:join|we at|hiring at|from)\s+([A-Z][A-Za-z0-9&\-\.\'\s]{2,40})", body_text, re.IGNORECASE)
    if body_match:
        print("extract company done by method 2")
        return body_match.group(1).strip()

    for tag in soup.find_all("meta"):
        if tag.get("property") == "og:site_name":
            print("extract company done by method 3")
            return tag.get("content", "").strip()

    if soup.title and "|" in soup.title.text:
        print("extract company done by method 4")
        return soup.title.text.split("|")[-1].strip()

    return soup.title.text.strip() if soup.title else "Unknown"

# 🧠 Node: Detect job page
async def detect_node(state: GraphState) -> GraphState:
    """
    Uses an AI call with a Pydantic schema to accurately determine if a URL is a job page.
    """
    print("\n--- Node: Detect Job Page ---")
    url = state["job_url"]
    html_content = await scrape_with_playwright(url)
    if not html_content:
        print("  Detection failed: Could not scrape page.")
        return {**state, "is_job_page": False}

    cleaned_text = clean_text(html_content)
    
    # Using an LLM with a structured output schema for reliable detection
    detector_llm = llm.with_structured_output(JobPageAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at analyzing webpages. Your task is to determine if the provided text from a URL points to a specific job posting. Look for keywords like 'Apply Now', lists of 'Responsibilities', and 'Qualifications'. Distinguish between general career pages and pages for a single job."),
        ("human", "Analyze the following text from the URL '{url}' and provide your analysis:\n\n---\n{text}\n---")
    ])
    
    chain = prompt | detector_llm
    analysis_result = await chain.ainvoke({"url": url, "text": textwrap.shorten(cleaned_text, 15000)})
    
    print(f"  AI Detection Result: {analysis_result.is_job_posting} (Confidence: {analysis_result.confidence_score:.2f})")
    print(f"  Reasoning: {analysis_result.reasoning}")

    return {
        **state,
        "is_job_page": analysis_result.is_job_posting and analysis_result.confidence_score > 0.6,
        "scraped_html_text": cleaned_text,
        "job_title": analysis_result.detected_job_title,
        "company_name": analysis_result.detected_company_name,
    }

# 🔍 Node: Search
async def search_node(state: GraphState) -> GraphState:
    """
    Performs targeted web searches to gather additional context about the company and job.
    """
    print("\n--- Node: Search for Additional Info ---")
    company_name = state["company_name"]
    job_title = state["job_title"]

    def run_search(query):
        results = search_tool.run(query)
        return "\n".join([r.get("content", "") for r in results]) if isinstance(results, list) else str(results)

    # Run searches in parallel for efficiency
    company_query = f'"{company_name}" company mission, vision, and values'
    linkedin_query = f'"{company_name}" site:linkedin.com/company'
    
    company_results, linkedin_results = await asyncio.gather(
        asyncio.to_thread(run_search, company_query),
        asyncio.to_thread(run_search, linkedin_query)
    )
    
    print("  Web searches completed.")
    return {
        **state,
        "search_results": company_results,
        "linkedin_results": linkedin_results,
    }



# 📝 Prepare Prompt + Log to File
def research_agent_node(state: GraphState) -> GraphState:
    """
    The main agent that synthesizes all gathered information into the final structured output.
    """
    print("\n--- Node: Research Agent ---")
    
    # Create a new agent that is bound to our desired output format
    structured_llm_agent = llm.with_structured_output(CompanyResearchOutput)
    
    system_prompt = """You are an elite company and job research assistant. Your goal is to produce a structured, comprehensive summary based on the provided data.

    **Rules:**
    1.  **Extract Real Information:** Do not use placeholders. If critical information is missing, state that it could not be found.
    2.  **Verify Company Name:** The provided company name might be a job board (like Wellfound, Greenhouse). Cross-reference the content to find the true hiring company.
    3.  **Synthesize, Don't Just Copy:** Combine information from the scraped content and search results to create a coherent summary.
    4.  **Complete All Fields:** Fill out every field in the requested `CompanyResearchOutput` format as completely as possible."""
    
    human_template = """Please conduct the research based on the following data:

    **Job URL:** {job_url}
    **Detected Job Title:** {job_title}
    **Detected Company Name:** {company_name}

    **--- Scraped Page Content ---**
    {scraped_html_text}

    **--- Company Search Results (Mission/Vision) ---**
    {search_results}

    **--- LinkedIn Search Results ---**
    {linkedin_results}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    agent_chain = prompt | structured_llm_agent
    
    final_output = agent_chain.invoke(state)
    
    print("  Agent has generated the final structured output.")
    return {**state, "final_output": final_output}

# 🎯 Capture Output
async def capture_output(state):
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'content'):
            state["final_output"] = msg.content
            break
    print("Captured output")
    return state

# 📊 Graph State
class GraphState(TypedDict):
    job_url: str
    is_job_page: bool
    scraped_html_text: str
    job_title: str
    company_name: str
    search_results: str
    job_search_results: str
    linkedin_results: str
    messages: List
    final_output: CompanyResearchOutput

# Minimal GraphState for detection only
class DetectOnlyState(TypedDict):
    job_url: str
    is_job_page: bool
    scraped_html: str
    job_title: str
    company_name: str

# Build detection-only graph
def build_detect_only_graph():
    graph = StateGraph(DetectOnlyState)
    graph.add_node("detect", detect_node)
    graph.set_entry_point("detect")
    graph.add_edge("detect", END)
    return graph.compile()


# 🧠 Build Graph
def build_graph():
    """
    Builds the LangGraph agent graph.
    """
    graph = StateGraph(GraphState)
    graph.add_node("detect", detect_node)
    graph.add_node("search", search_node)
    graph.add_node("research_agent", research_agent_node)
    
    graph.set_entry_point("detect")

    # Conditional edge: If it's a job page, proceed to search. Otherwise, end.
    graph.add_conditional_edges(
        "detect",
        lambda state: "search" if state["is_job_page"] else END,
        {"search": "search", END: END}
    )
    graph.add_edge("search", "research_agent")
    graph.add_edge("research_agent", END)
    
    return graph.compile()

# 🚀 Runner
async def run_job_research(job_url: str) -> CompanyResearchOutput | None:
    """
    The main execution function to run the job research agent.
    """
    print(f"\n Initializing job research for: {job_url}")
    graph = build_graph()
    
    initial_state = GraphState(
        job_url=job_url,
        is_job_page=False,
        scraped_html_text="",
        job_title="",
        company_name="",
        search_results="",
        job_search_results="",
        linkedin_results="",
        messages=[],
        final_output=None
    )
    
    final_state = await graph.ainvoke(initial_state)
    
    if final_state and final_state.get("final_output"):
        print("\n Research complete. Final output generated.")
        return final_state["final_output"]
    else:
        print("\n Research ended. The provided URL was not identified as a job posting page.")
        return None
