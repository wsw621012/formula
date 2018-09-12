#Download base image ubuntu 16.04
FROM ubuntu:16.04

# Install necessary packages

RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y python-pip


RUN pip install --upgrade pip
RUN pip install pillow flask-socketio eventlet opencv-python

# Copy the current directory contents into the container at /app
RUN mkdir /app
COPY sample_bot_car_TI6.py /app

# Set the working directory to /app
WORKDIR /app

# Run formula_bot.py when the container launches, u should replace with ur program
ENTRYPOINT ["python","sample_bot_car_TI6.py"]
