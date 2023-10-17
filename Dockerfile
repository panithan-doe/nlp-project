FROM python:3.7.9
COPY . .
RUN pip install -r requirements.txt
CMD ["python","wsgi.py"]