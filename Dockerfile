FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
# optional: copy a tiny healthcheck
RUN useradd -m appuser
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
