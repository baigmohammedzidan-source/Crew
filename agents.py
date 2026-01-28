
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.agents import SequentialAgent,ParallelAgent
from google.genai import types


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


Micro_agent= LlmAgent(
    name="Micro",
    model=Gemini(model="gemini-2.5-flash-lite",
                    retry_options=retry_config),
    description="A simple financial agent that analyze data and gives and answer for the next minute to days.",
    instruction="You are a expert business analist,Your task is to go throught the data using google_search tool and give your opinon in yes or no(This is for upcoming minutes to days),Give answer in short and structured manner give maximun 5 bullit points to justify your answer,The bullit point should be one liners",
    tools=[google_search],
    output_key="Mirco_data",
)

Meso_agent= LlmAgent(
    name="Meso",
    model=Gemini(model="gemini-2.5-flash-lite",
                    retry_options=retry_config),
    description="A simple agent that can analyze data and gives and answer for upcoming days to weeks.",
    instruction="You are a expert business analist,Your task is to go throught the data using google_search tool and give your opinon in yes or no(This is for few upcoming days to weeks),Give answer in short and structured manner preferabily in less than 30 words,Use 5 bullit points wherever possible",
    tools=[google_search],
    output_key="Meso_data",
)

Macro_agent= LlmAgent(
    name="Macro",
    model=Gemini(model="gemini-2.5-flash-lite",
                    retry_options=retry_config),
    description="A simple agent that can analyze data and gives and answer for upcoming weeks to month.",
    instruction="You are a expert business analist,Your task is to go throught the data using google_search tool and give your opinon in yes or no(This is for upcoming weeks this should be less than decades),Give answer in short and structured manner give maximun 5 bullit points to justify your answer,The bullit point should be one liners",
    tools=[google_search], 
    output_key="Macro_data",
)

Meta_agent= LlmAgent(
    name="Meta",
    model=Gemini(model="gemini-2.5-flash-lite",
                    retry_options=retry_config),
    description="A simple agent that can analyze data and gives and answer for upcoming months to year.",
    instruction="You are a expert business analist,Your task is to go throught the data using google_search tool and give your opinon in yes or no(This is for upcoming months to year),Give answer in short and structured manner give maximun 5 bullit points to justify your answer,The bullit point should be one liners.",
    tools=[google_search],
    output_key="Meta_data",)

Altro_agent= LlmAgent(
    name="Altro",
    model=Gemini(model="gemini-2.5-flash-lite",
                    retry_options=retry_config),
    description="You are an expert financial agent who knows a lot about stocks and trading",
    instruction="You are an expert financial agent who knows a lot about stocks and trading,Use google_search tool and provide alternate stocks which the user can go with. Give 5 alternative stocks with 1 reason why you feel they are better than what stock user has provided with.keep it well structured and leave a line after every stock suggestion ",
    tools=[google_search],
    output_key="Altro_data", )

# The AggregatorAgent runs *after* the parallel step to synthesize the results.
review_agent = LlmAgent(
    name="ReviewAgent",
    model=Gemini(model="gemini-2.5-flash-lite",
                    retry_options=retry_config),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a single executive summary:


    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all four reports. The final summary should be around 50 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

parallel_agent = ParallelAgent(
    name="Parallel",
    sub_agents=[Micro_agent,Meso_agent,Macro_agent,Meta_agent])

root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_agent,review_agent,Altro_agent],)



