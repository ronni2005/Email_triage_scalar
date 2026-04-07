import uvicorn
from fastapi import FastAPI
from env import create_env
from agent import SmartEmailAgent
from grader import evaluate_agent
from models import EmailAction, StepResult

app = FastAPI()
env = create_env(task="hard")
agent = SmartEmailAgent()

@app.get("/")
def home():
    return {"message": "Email Triage is LIVE. Go to /test to see efficiency!"}

@app.get("/test")
def test():
    
    score = evaluate_agent(env, agent, episodes=10)
    return {"efficiency_score": score}

@app.post("/reset", response_model=StepResult)
def reset():
    return env.reset()

@app.post("/step", response_model=StepResult)
def step(action_input: EmailAction):
   
    res = env.step(action_input)
    print(f"ACTION: {action_input.action} | REWARD: {res.reward}")
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)