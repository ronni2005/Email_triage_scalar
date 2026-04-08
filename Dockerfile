FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn numpy openai pydantic openenv-core
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
CMD ["python", "-u", "inference.py"]
