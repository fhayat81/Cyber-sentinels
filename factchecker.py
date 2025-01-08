import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

model1 = ChatGroq(model="llama3-8b-8192")

def fact_check(text):
    print("Starting Fact check...")
    claims = text.split('.')
    claims = list(filter(lambda x: x != "", claims))

    length = len(claims)
    size = int(length/50)

    def concatenate_every_three(strings):
        # Use a list comprehension to group and concatenate every 3 strings
        result = ['. '.join(strings[i:i+3]) for i in range(0, len(strings), size)]
        return result

    claims = concatenate_every_three(claims)

    count = 0
    score = 0

    for claim in claims:
        answer = model1.invoke([HumanMessage(content=f"How factually correct is this sentence on scale of 0 to 1 without any explaination till third decimal place: {claim}")])
        #print(f"{answer.content}\n")
        try:
          score += float(answer.content)
          count += 1
        except ValueError:
          score += 0.0


    print(f"Factual correctness score: {(score/count)*100:.2f}%")
    return (score/count)*100