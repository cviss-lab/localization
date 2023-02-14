FROM python:3.9-slim

# open3d requires cmake 3.20, which isn't available on debian bullseye (the os
# python:3.9 is based on). So we have to use the backport repository
RUN echo "deb http://deb.debian.org/debian bullseye-backports main" > /etc/apt/sources.list.d/backports.list
# Some of the c libraries requires compilation
RUN apt-get update && apt-get install -y --no-install-recommends -t bullseye-backports \
    cmake git build-essential \
    weasyprint \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# install dependencies
RUN pip install --upgrade pip

COPY ./localization/api/requirements.txt .

RUN pip install -r requirements.txt

# copy project
COPY ./localization/libs /app/libs
COPY ./localization/api /app/api

WORKDIR /app/api
CMD ["python3", "app.py"]
