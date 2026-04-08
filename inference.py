import os
import sys
from fastapi import FastAPI
from env import create_env
from agent import SmartEmailAgent
from grader import evaluate_agent
from models import EmailAction, StepResult


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

steps_count = 0
total_reward = 0

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
    global steps_count, total_reward
    steps_count = 0
    total_reward = 0
    
    print("[START] task=email_triage", flush=True)
    sys.stdout.flush()
    return env.reset()

@app.post("/step", response_model=StepResult)
def step(action_input: EmailAction):
    global steps_count, total_reward
    
    res = env.step(action_input)
    
    steps_count += 1
    total_reward += float(res.reward)
    
    print(f"[STEP] step={steps_count} reward={float(res.reward)}", flush=True)
    
    if res.done:
        print(f"[END] task=email_triage score={float(total_reward)} steps={steps_count}", flush=True)
    
    return res
