
FROM centos:7

# Install necessary packages

RUN yum install -y epel-release
RUN yum install -y python-pip
RUN yum install -y opencv-python

RUN pip install pillow flask-socketio eventlet

# Copy the current directory contents into the container at /app
RUN mkdir /app
RUN mkdir /app/logs
COPY sample_bot_car_TI6.py /app

# Set the working directory to /app
WORKDIR /app

# Run formula_bot.py when the container launches, u should replace with ur program
ENTRYPOINT ["python","sample_bot_car_TI6.py"]
