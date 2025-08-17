# Library Assistant
Library Assistant built with the OpenAI Agents SDK. 

Supports: book search, availability checks for members, and library timings with input guardrails.

**Steps:**

- User Context → Pydantic model with name and member_id.
- Guardrail Agent → Stops non-library queries.
- Input Guardrail Function → Uses guardrail agent.
- Member Check Function → Allows availability tool only if user is valid.

**Function Tools:**

- Search Book Tool → Returns if the book exists.
- Check Availability Tool → Returns how many copies are available.
- Dynamic Instructions → Personalize based on user name.
- Library Agent → Add tools, guardrails, and model settings.
- Book Database → Store book names and copies in a Python dictionary; tools use this data.
- Multiple Tools Handling → Make sure agent can search and check availability in one query.
