from agents import Agent

a = Agent(name="test", instructions="say hi")
print("Has copy:", hasattr(a, "copy"))