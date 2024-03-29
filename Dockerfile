FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "/app/lib/main.py"]