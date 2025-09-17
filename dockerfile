FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "src/api.py"]
Build & run:
docker build -t cust-personality:latest .
docker run -d -p 5000:5000 cust-personality:latest