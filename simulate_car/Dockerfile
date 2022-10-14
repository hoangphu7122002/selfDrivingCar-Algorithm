FROM python:3.8

COPY requirements.txt .
RUN pip install -r requirements.txt && \
	rm requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


# EXPOSE 11000

COPY . .

CMD ["python","./test_final.py"]