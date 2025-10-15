import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict
from flask import Flask, render_template, request
import markdown
import requests

# --- 0. Load API Keys and Basic Setup ---
load_dotenv()
app = Flask(__name__) # Initialize the Flask app

# Verify that keys are loaded
assert os.getenv("GOOGLE_API_KEY"), "Please set your GOOGLE_API_KEY in the .env file."
assert os.getenv("TAVILY_API_KEY"), "Please set your TAVILY_API_KEY in the .env file."
assert os.getenv("NEWS_API_KEY"), "Please set your NEWS_API_KEY in the .env file."
assert os.getenv("PEXELS_API_KEY"), "Please set your PEXELS_API_KEY in the .env file." # <-- 2. ADD PEXELS KEY CHECK

# --- START: All your existing LangGraph agent code ---

# Import LangChain and LangGraph components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import wikipedia
from newsapi import NewsApiClient
from langgraph.graph import StateGraph, END

# 1. Define the Agent State
class AgentState(TypedDict):
    topic: str
    sub_topics: List[str]
    search_results: Dict[str, List[str]]
    report: str
    error: str

# 2. Create the Information Gathering Tools
tavily_tool = TavilySearch(max_results=4)

def search_wikipedia(query: str) -> str:
    try:
        content = wikipedia.summary(query, sentences=3, auto_suggest=False)
        return f"Wikipedia Summary for '{query}':\n{content}"
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {e}"

newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
def search_news(query: str) -> str:
    try:
        top_headlines = newsapi.get_top_headlines(q=query, language='en', page_size=5)
        articles = top_headlines.get('articles', [])
        if not articles: return f"No recent news found for '{query}'."
        results = [f"Title: {a['title']}\nSource: {a['source']['name']}\nDescription: {a['description']}" for a in articles]
        return f"News Articles for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error searching NewsAPI for '{query}': {e}"

# 3. Define the Graph Nodes
# NOTE: The model name was updated to a valid one. "gemini-2.0-flash" does not exist as of my last update.
# Please use "gemini-1.5-flash-latest" or another available model.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def plan_research_node(state: AgentState) -> dict:
    print("--- 📝 Planning Research ---")
    prompt = ChatPromptTemplate.from_template(
        """You are a master research planner. Your goal is to break down a broad topic into a set of 3-5 specific,
        researchable sub-topics. Topic: {topic}. Provide a JSON object with a single key "sub_topics" containing a list of these sub-topics."""
    )
    planner_chain = prompt | llm | JsonOutputParser()
    try:
        result = planner_chain.invoke({"topic": state["topic"]})
        return {"sub_topics": result["sub_topics"]}
    except Exception as e:
        return {"error": f"Failed to plan research: {e}"}

def conduct_search_node(state: AgentState) -> dict:
    print("--- 🔍 Conducting Search ---")
    all_results = {}
    sub_topics = state["sub_topics"]
    
    # Ensure sub_topics is a list to prevent iteration errors
    if not isinstance(sub_topics, list):
        return {"error": "Sub-topics are not in a valid list format."}

    for sub_topic in sub_topics:
        # Ensure sub_topic is a string before using it as a query
        if not isinstance(sub_topic, str):
            print(f"--- Skipping invalid sub_topic (not a string): {sub_topic} ---")
            continue
            
        print(f"--- Searching for: {sub_topic} ---")
        
        # Initialize a list to hold string-based results
        string_results = []

        # 1. Tavily Search
        try:
            tavily_raw_results = tavily_tool.invoke(sub_topic)
            # Format the dictionary output into a single string
            formatted_tavily = "\n\n".join(
                [f"URL: {res.get('url', 'N/A')}\nContent: {res.get('content', 'N/A')}" for res in tavily_raw_results]
            )
            string_results.append(f"Tavily Search Results:\n{formatted_tavily}")
        except Exception as e:
            string_results.append(f"Error during Tavily search: {e}")

        # 2. Wikipedia Search
        string_results.append(search_wikipedia(sub_topic))

        # 3. News Search
        string_results.append(search_news(sub_topic))
        
        # Store the list of strings for this sub-topic
        all_results[sub_topic] = string_results
        
    return {"search_results": all_results}

def synthesize_report_node(state: AgentState) -> dict:
    print("--- ✍️ Synthesizing Report ---")
    formatted_results = ""
    for sub_topic, results in state["search_results"].items():
        formatted_results += f"--- Research on: {sub_topic} ---\n"
        for result in results:
            formatted_results += str(result) + "\n\n"
    prompt = ChatPromptTemplate.from_template(
        """You are a professional research analyst. Your task is to write a comprehensive,
        well-structured report on a given topic using the provided research material.
        Synthesize the information into a coherent narrative.
        **Topic:** {topic}
        **Collected Research Material:** {research_material}
        **Your Final Report:**"""
    )
    synthesis_chain = prompt | llm | StrOutputParser()
    try:
        report = synthesis_chain.invoke({"topic": state["topic"], "research_material": formatted_results})
        return {"report": report}
    except Exception as e:
        return {"error": f"Failed to synthesize report: {e}"}

def get_image_for_topic(topic: str) -> str | None:
    """Fetches a relevant image URL from Pexels for a given topic."""
    try:
        url = f"https://api.pexels.com/v1/search"
        headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
        params = {"query": topic, "per_page": 1, "page": 1}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Raises an exception for bad status codes
        
        data = response.json()
        if data.get("photos"):
            # Return the URL of the medium-sized photo
            return data["photos"][0]["src"]["large"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from Pexels: {e}")
    
    return None # Return None if anything fails

# 4. Assemble and Compile the Graph
graph = StateGraph(AgentState)
graph.add_node("planner", plan_research_node)
graph.add_node("searcher", conduct_search_node)
graph.add_node("synthesizer", synthesize_report_node)
graph.set_entry_point("planner")
graph.add_edge("planner", "searcher")
graph.add_edge("searcher", "synthesizer")
graph.add_edge("synthesizer", END)
research_agent = graph.compile()

# --- 5. Define Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def research():
    topic = request.form.get('topic')
    if not topic:
        return "Error: Please provide a topic.", 400
    
    # <-- 3. CALL THE NEW IMAGE FUNCTION ---
    image_url = get_image_for_topic(topic)
    
    print(f"🚀 Starting research on: '{topic}'")
    initial_input = {"topic": topic}
    final_state = research_agent.invoke(initial_input)
    
    if final_state.get("error"):
        report_content = f"An Error Occurred:\n\n{final_state['error']}"
    else:
        report_content = final_state['report']

    html_report = markdown.markdown(report_content, extensions=['fenced_code', 'tables'])
        
    # <-- 4. PASS IMAGE URL TO TEMPLATE ---
    return render_template('report.html', topic=topic, report=html_report, image_url=image_url)


# --- 6. Run the Flask App ---

if __name__ == "__main__":
    app.run(port=5001,debug=True)