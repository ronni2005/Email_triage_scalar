import os
import sys
import uvicorn
import numpy as np
from fastapi import FastAPI
from env import create_env
from agent import SmartEmailAgent
from models import EmailAction, StepResult


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TASK_NAME = "email_triage"
BENCHMARK = "hard"


steps_count = 0
total_reward = 0
rewards_list = []

app = FastAPI()
env = create_env(task=BENCHMARK)
agent = SmartEmailAgent()

@app.get("/")
def home():
    return {"status": "LIVE"}

@app.post("/reset", response_model=StepResult)
def reset():
    global steps_count, total_reward, rewards_list
    steps_count = 0
    total_reward = 0
    rewards_list = []
    
    
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    return env.reset()

@app.post("/step", response_model=StepResult)
def step(action_input: EmailAction):
    global steps_count, total_reward, rewards_list
    
    res = env.step(action_input)
    
    steps_count += 1
    reward = float(res.reward)
    total_reward += reward
    rewards_list.append(reward)

    
    done_val = "true" if res.done else "false"
    
    action_str = f"classify({int(action_input.action)})"
    
    
    print(f"[STEP] step={steps_count} action={action_str} reward={reward:.2f} done={done_val} error=null", flush=True)
    
    if res.done:
       
        final_score = float(np.tanh(total_reward / 20))
        final_score = min(max(final_score, 0.0), 1.0) # Clamp to [0, 1]
        
        success_val = "true" if final_score >= 0.1 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
        
      
        print(f"[END] success={success_val} steps={steps_count} score={final_score:.2f} rewards={rewards_str}", flush=True)
        
    return res

if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
