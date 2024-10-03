FROM python:3.9-slim

WORKDIR /code/deployment/api

COPY models ../../../models
COPY code/deployment/api/requirements.txt .
COPY code/deployment/api/app.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]