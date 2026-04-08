import os, sys, numpy as np, uvicorn
from fastapi import FastAPI, BackgroundTasks
from env import create_env
from agent import SmartEmailAgent

app = FastAPI()
TASK_NAME = "email_triage"
BENCHMARK = os.getenv("BENCHMARK", "hard")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

@app.get("/")
def health(): return {"status": "LIVE"}

@app.post("/reset")
def reset(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_inference_loop)
    return {"status": "starting_inference"}

def run_inference_loop():
    try:
        env = create_env(task=BENCHMARK)
        agent = SmartEmailAgent()
        obs = env.reset()
        
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        steps, total_reward, rewards = 0, 0, []
        done = False
        
        while not done:
            action_val = agent.predict(obs)
            res = env.step(action_val)
            steps += 1
            reward = float(res.reward)
            total_reward += reward
            rewards.append(reward)
            obs = res.observation
            done = res.done

            print(f"[STEP] step={steps} action=classify({int(action_val)}) reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        score = min(max(float(np.tanh(total_reward / 20)), 0.0), 1.0)
        success = "true" if score >= 0.1 else "false"
        print(f"[END] success={success} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="error")
