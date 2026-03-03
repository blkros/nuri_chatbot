FROM python:3.11-slim

# System dependencies: LibreOffice (HWP→PDF), poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-nanum \
    fonts-nanum-extra \
    fonts-nanum-coding \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
