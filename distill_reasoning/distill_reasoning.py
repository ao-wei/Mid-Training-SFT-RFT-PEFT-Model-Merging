import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def request_chat(query, is_math=False):
    if is_math:
        PROMPT = r"请一步步推理，并把最终答案放到 \boxed{}。"
    else:
        PROMPT = ""
    try:
        messages = [
            {"role": "user", "content":  PROMPT + query}
        ]
        response = llm_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return reasoning_content, content
        
    except:
        return "", ""

# 测试 1
query = "计算矩阵A=[[1,2,3],[4,5,6]]的秩。"
reasoning, answer = request_chat(query, is_math=True)
print("reasoning and answer: ")
print("="*50)
print(reasoning)
print("="*50)
print(answer)