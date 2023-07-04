FROM python:3.11.3-slim

WORKDIR /app

COPY ui.py inference.py model.pth model.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "ui.py"]