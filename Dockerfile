FROM python:3.12-slim

WORKDIR /app

# Install all dependencies from requirements.txt so a rebuild stays in sync.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]
