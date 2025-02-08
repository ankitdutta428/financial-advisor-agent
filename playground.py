from phi.agent import Agent 
from phi.model.groq import Groq 
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app


from dotenv import load_dotenv
load_dotenv()


# WebSearch Agent
websearch_agent = Agent(
    name = "Web-Search-Agent", 
    role = "Search the web for the information",
    model = Groq(id="llama-3.3-70b-versatile"), 
    tools = [DuckDuckGo()], 
    instructions = ['Always include the sources'],
    show_tool_calls= True, 
    markdown=True,
    
)


finance_agent = Agent(
    name="Finance Agent",
    role = "Find the finance information!",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data and also display the latest news"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True, 
)


multi_agent = Agent(
    team = [websearch_agent, finance_agent], 
    model = Groq(id = "llama-3.3-70b-versatile"),
    instructions = ["Always include the sources", "Use tables to show the data"],
    show_tool_calls=True, 
    markdown = True,
    debug_mode=True,
)


app = Playground(agents=[finance_agent, websearch_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
