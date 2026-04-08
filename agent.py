import os
from openai import OpenAI

class SmartEmailAgent:
    def __init__(self):
      
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    def predict(self, observation):
        try:
           
            prompt = f"Given this email state, choose action (0: Archive, 1: Reply, 2: Forward, 3: Flag): {observation}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5
            )
        
            res_text = response.choices[0].message.content.strip()
            return int(''.join(filter(str.isdigit, res_text)) or 0)
        except Exception:
            return 0 
