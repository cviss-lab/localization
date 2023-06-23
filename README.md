# Localization

Server for image-based localization

## Prequsites:

- [docker](https://docs.docker.com/engine/install/) (makes sure to run the
  [linux post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)
- [python](https://www.python.org/downloads/) 3.9+
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Running locally

- `docker compose up --build` to build and run the containers (add the `-d`
  flag if you want it to run in detached mode, aka in the background). The app
  can then be accessed via `localhost:5000`
- `docker compose down` to stop the containers
- `docker compose exec api bash` to exec into the container

## Project architecture

The app is composed of a backend (`./api`) written in Python using the Flask
framework.

Previous projects should be stored in `$HOME/datasets` under `project_1`, `project_2`, etc...

Each project has the following architecture
```
.
├── ...
├── project_1            # Project
│   ├── rgb              # RGB reference images
│      ├── 1.png         
│      |── 2.png                            
│      └── ...                            
│   ├── depth            # depth reference images
│      ├── 1.png         
│      |── 2.png                            
│      └── ...         
│   |── poses.csv        # image pose of reference camera when each image was captured
│   └── intrinsics.json  # camera intrinsics of reference camera
└── ...
```
## Sample Data
Download this file, unzip it, then save the folder as `~/datasets/project_1`.

Google drive link: [sample dataset](https://drive.google.com/file/d/1yosTKMK7qdtyPpteEc6Dljv_n4YOv1xK/view?usp=sharing)

## API Examples
Loading project: 
```
curl http://localhost:5000/api/v1/project/1/load
```
Providing query camera intrinsics (assuming query camera is different from reference camera):
```
curl -X POST -H "Content-Type: application/json"  -d @<path-to-intrinsics> http://localhost:5000/api/v1/project/1/intrinsics
```
Localizing query image:
```
curl -X POST -F image=@<path-to-img> http://localhost:5000/api/v1/project/1/localize
```
Sending raw data in zip file to localization server
```
curl -X POST -H "Content-Type: application/octet-stream" -F "data=@<path-to-zip>" http://localhost:5000/api/v1/project/1/upload
```
