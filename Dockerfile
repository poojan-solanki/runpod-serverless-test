FROM python:3.11
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY rp_handler.py /app/
# Start the container
CMD ["python3", "-u", "rp_handler.py"]