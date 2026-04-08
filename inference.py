import uvicorn
import os
from fastapi import FastAPI
from env import create_env
from agent import SmartEmailAgent
from grader import evaluate_agent
from models import EmailAction, StepResult


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")


app = FastAPI()
env = create_env(task="hard")
agent = SmartEmailAgent()

@app.get("/")
def home():
    return {"message": "Email Triage is LIVE. Go to /test to see efficiency!"}

@app.get("/test")
def test():
    try:
        score = evaluate_agent(env, agent, episodes=10)
        return {"efficiency_score": score}
    except Exception as e:
        return {"error": str(e)}

@app.post("/reset", response_model=StepResult)
def reset():
    print("START") 
    return env.reset()

@app.post("/step", response_model=StepResult)
def step(action_input: EmailAction):
    res = env.step(action_input)
    
    # Requirement: STEP <action> <reward> log
    print(f"STEP {action_input.action} {res.reward}")
    
    if res.done:
        print("END") 
        
    return res


