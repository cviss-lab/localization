FROM python:3.9-slim

# open3d requires cmake 3.20, which isn't available on debian bullseye (the os
# python:3.9 is based on). So we have to use the backport repository
RUN echo "deb http://deb.debian.org/debian bullseye-backports main" > /etc/apt/sources.list.d/backports.list
# Some of the c libraries requires compilation
RUN apt-get update && apt-get install -y --no-install-recommends -t bullseye-backports \
    git build-essential \
    wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# build open3d
COPY ./localization/api/requirements.txt .
COPY ./localization/dockerfiles/api/build_open3d.sh /
# RUN /build_open3d.sh

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# download and rename models
COPY ./localization/dockerfiles/api/download_models.sh download_models.sh 
RUN chmod +x download_models.sh
RUN ./download_models.sh

# copy project
COPY ./localization/libs /app/libs
COPY ./localization/api /app/api
COPY ./localization/config /app/config

# setup endpoint
WORKDIR /app/api
CMD ["python3", "app.py"]
ENV OPEN3D_CPU_RENDERING true
