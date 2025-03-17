FROM python:3.11
WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY rp_handler.py /
# Start the container
CMD ["python3", "-u", "rp_handler.py"]