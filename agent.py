import os
import re
from openai import OpenAI

class SmartEmailAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = os.getenv("MODEL_NAME")

    def predict(self, observation):
        try:
         
            prompt = f"""
            You are an Email Triage Expert. Analyze the state and pick the best action.
            State: {observation}
            
            Actions:
            0: Archive (Spam/Neutral)
            1: Reply (High Priority/Question)
            2: Forward (Technical/Team task)
            3: Flag (Urgent/Action needed)

            Reply with ONLY the number (0, 1, 2, or 3).
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                temperature=0.1
            )
            
            res_text = response.choices[0].message.content.strip()
        
            digit_match = re.search(r'\d', res_text)
            if digit_match:
                action = int(digit_match.group())
                return action if action in [0, 1, 2, 3] else 0
            return 0
        except Exception as e:
            return 0
