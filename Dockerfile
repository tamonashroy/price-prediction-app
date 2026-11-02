# Runs the Streamlit dashboard. Pass DATABASE_URL at run time: docker run --env-file .env -p 8501:8501 <image>
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY pricepredict ./pricepredict
COPY assets ./assets

EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py"]
