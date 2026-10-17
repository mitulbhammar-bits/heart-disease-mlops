FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-serving-lock.txt .
RUN pip install --no-cache-dir -r requirements-serving-lock.txt && useradd --uid 10001 --create-home appuser
COPY heart_risk ./heart_risk
COPY models ./models
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')" || exit 1
CMD ["uvicorn", "heart_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
