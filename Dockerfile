FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY level2.py .

EXPOSE 8000

CMD ["python", "-m", "fastapi", "run", "level2.py", "--host", "0.0.0.0", "--port", "8000"]
