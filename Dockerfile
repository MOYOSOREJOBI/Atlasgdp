FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OFFLINE_MODE=1 \
    ARTIFACT_ROOT=/app/artifacts

WORKDIR /app

COPY pyproject.toml README.md requirements.txt LICENSE ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY Makefile ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /root/.streamlit \
    && printf "[server]\nheadless = true\naddress = \"0.0.0.0\"\nport = 8501\n\n[browser]\ngatherUsageStats = false\n" > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py"]
