import uvicorn
from fastapi import FastAPI
from inference import run_inference # Import from root

app = FastAPI()

@app.get("/")
def health(): return {"status": "LIVE"}

@app.post("/reset")
def reset():
   
    return {"message": "Environment Reset"}

@app.post("/step")
def step(action: dict):
    return {"message": "Step taken"}

def main():
   
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

if __name__ == "__main__":
    main()
