FROM python:3.11.3-slim

WORKDIR /app

COPY ui.py inference.py model.pth model.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port", "8501"]