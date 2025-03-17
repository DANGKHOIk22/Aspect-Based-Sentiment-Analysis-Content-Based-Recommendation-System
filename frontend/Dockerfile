FROM python:3.11.9-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

EXPOSE 3000

CMD ["streamlit", "run", "main.py", "--server.port=3000", "--server.address=0.0.0.0"]
