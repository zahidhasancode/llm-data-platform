FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 \
    fastapi==0.115.6 uvicorn==0.34.0
WORKDIR /app
COPY src /app/src
USER 1000
ENV HF_HOME=/tmp/hf
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
