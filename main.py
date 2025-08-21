import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, input_guardrail, RunContextWrapper, ModelSettings, enable_verbose_stdout_logging
from agents.run import RunConfig
from pydantic import BaseModel, Field
from typing import Dict, Optional

# enable_verbose_stdout_logging()

# --- USER CONTEXT ---
class UserContext(BaseModel):
    name: str = Field(..., description="Name of the user")
    member_id: Optional[str] = Field(None, description="Membership ID of the user (None if not a member)")

# --- ENVIRONMENT VARIABLES ---
load_dotenv()

set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model_gemini = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model_gemini,
    model_provider=external_client,
    tracing_disabled=True
)

# --- BOOKS DATABASE ---
BOOK_DB: Dict[str, int] = {
    "Clean Code": 2,
    "The Pragmatic Programmer": 0,
    "Introduction to Algorithms": 3,
    "Design Patterns": 1,
    "Deep Learning": 4,
}

LIBRARY_TIMINGS: Dict[str, str] = {
    "monday": "9:00 – 19:00",
    "tuesday": "9:00 – 19:00",
    "wednesday": "9:00 – 19:00",
    "thursday": "9:00 – 19:00",
    "friday": "9:00 – 19:00",
    "saturday": "10:00 – 16:00",
    "sunday": "10:00 – 14:00",
}

VALID_MEMBERS = {"M-1001", "M-2002", "M-3003"}

# --- GUARDRAIL AGENT ---
class GuardrailAgent(Agent):    
    def __init__(self):
        super().__init__(
            name="guardrail-agent",
            instructions=(
                "You are a strict gatekeeper for a library assistant. "
                "If the user's message is about books, availability, membership, or library timings, respond with EXACTLY 'ALLOW'. "
                "For anything else (e.g., sports, politics, finance, chit-chat), respond with EXACTLY 'BLOCK'. "
                "No extra words."
            ),
            model=model_gemini,
        )

    def copy(self):
        return super().copy()


guardrail_agent = GuardrailAgent()

# --- INPUT GUARDRAIL ---
@input_guardrail
async def library_input_guardrail(ctx: RunContextWrapper, user_message: str, tool_call = None) -> None:
    result = await Runner.run(guardrail_agent, user_message)
    if result.final_output.strip().upper() != "ALLOW":
        raise ValueError("❌ This assistant only answers library-related questions.")

# --- MEMBER CHECK ---
def is_member_allowed(ctx: RunContextWrapper) -> bool:
    uc: UserContext = ctx.context
    return bool(uc and uc.member_id and uc.member_id in VALID_MEMBERS)

# --- FUNCTION TOOLS ---
@function_tool
def search_book(ctx: RunContextWrapper, book_name: str) -> dict:
    return {
        "title": book_name,
        "in_catalog": book_name in BOOK_DB
    }

@function_tool(is_enabled=lambda ctx, _: is_member_allowed(ctx))
def check_availability(ctx: RunContextWrapper, book_name: str) -> dict:
    if book_name not in BOOK_DB:
        return {"title": book_name, "available_copies": 0, "note": "Not in catalog."}
    return {"title": book_name, "available_copies": BOOK_DB[book_name]}

@function_tool
def get_library_timings(ctx: RunContextWrapper, day: str) -> dict:
    key = day.lower()
    return {"day": day, "hours": LIBRARY_TIMINGS.get(key, "Unknown day.")}

# --- MAIN AGENT ---
library_agent = Agent(
    name="library-assistant",
    instructions=(
        "You are a helpful Library Assistant. "
        "You can: search for books, check availability for registered members, and provide library timings. "
        "Refuse non-library queries. Use tools when needed."
    ),
    model=model_gemini,
    model_settings=ModelSettings(
        temperature=1,
        tool_choice="auto",
    ),
    input_guardrails=[library_input_guardrail],
    tools=[search_book, check_availability, get_library_timings],
)

# --- HANDLE QUERY ---
async def handle_query(prompt: str, ctx: UserContext):
    personalized_prompt = (
        f"Hello {ctx.name}! (Member ID: {ctx.member_id}). "
        f"Please answer politely and help with library services. "
        f"User asked: {prompt}"
    )

    result = await Runner.run(
        library_agent,
        personalized_prompt,
        run_config=config,
        context=ctx
    )

    return result




# --- MAIN LOOP ---
async def run_loop():
    # Example user (change member_id=None to test guest access)
    user_context = UserContext(name="Omer", member_id="None")

    print("AI Bot:👋 Hello! I am a library assistant created by Muhammad Omer.")
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue

            result = await handle_query(prompt, user_context)
            print("\nAI Bot:", result.final_output)

        except KeyboardInterrupt:
            print("\nAI Bot:👋 Exiting. Thank you for using the bot!")
            break

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_loop())
