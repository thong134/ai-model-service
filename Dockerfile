# Sử dụng bản slim để nhẹ
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Cài đặt Python dependencies
# Ưu tiên cài đặt bản torch CPU-only để tiết kiệm RAM (giảm ~500MB+ so với bản thường)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Start the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "--workers", "1", "app.api:app"]
