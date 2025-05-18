FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install torch separately due to custom index, then install other dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
