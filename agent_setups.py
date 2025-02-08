from phi.agent import Agent 
from phi.model.groq import Groq 
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools


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


finance_agent.print_response("Summarize analyst recommendations for NVDA", stream = True)
