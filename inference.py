import os
import sys
import uvicorn
from fastapi import FastAPI
from env import create_env
from agent import SmartEmailAgent
from models import EmailAction, StepResult

# Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TASK_NAME = "email_triage"
BENCHMARK = "hard"


steps_count = 0
total_reward = 0
rewards_list = []
max_steps = 15 

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
    sys.stdout.flush()
    
    return env.reset()

@app.post("/step", response_model=StepResult)
def step(action_input: EmailAction):
    global steps_count, total_reward, rewards_list
    
    res = env.step(action_input)
    
    steps_count += 1
    current_reward = float(res.reward)
    total_reward += current_reward
    rewards_list.append(current_reward)
    

    done_val = "true" if res.done else "false"
    print(f"[STEP] step={steps_count} action={int(action_input.action)} reward={current_reward:.2f} done={done_val} error=null", flush=True)
    sys.stdout.flush()
    
    if res.done:
       
        score = min(max(total_reward / 10.0, 0.0), 1.0)
        success_val = "true" if score >= 0.1 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
        
        print(f"[END] success={success_val} steps={steps_count} score={score:.2f} rewards={rewards_str}", flush=True)
        sys.stdout.flush()
        
    return res

if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
