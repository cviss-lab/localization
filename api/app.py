import logging
import os
import flask
import typing
import shutil
import zipfile
from threading import Thread
import subprocess

import localization
from libs.utils.loader import *
from libs.utils.projection import *
from libs.utils.domain import *
import io
import tempfile

app = flask.Flask(__name__)
localizers: typing.Dict[int, localization.Localizer] = {}
intrinsics: typing.Dict[int, np.ndarray] = {}

def load_localizer(project_id: int) -> typing.Tuple[localization.Localizer]:
     
    home = os.environ.get('HOME')
    sample_data = os.path.join(f"{home}/datasets", f"project_{project_id}")
    if not os.path.exists(sample_data):
        return None
    l = LocalLoader(sample_data)
    loc = localization.Localizer(l)

    return loc

app.logger.info("API server ready")

@app.route('/api/v1/project/<int:project_id>/get_latest', methods=["GET"])
def get_latest(project_id):

    loc = localizers[project_id]
    
    # NOTE: anomaly detection happens here
    if loc.query_img is not None and loc.ret_img is not None:        
        I2 = cv2.hconcat([loc.query_img, loc.ret_img])
        I2 = cv2.resize(I2, (1280, 480))
    else:
        I2 = np.zeros((480, 1280, 3), dtype=np.uint8)

    data = cv2.imencode('.png', I2)[1].tobytes()
    resp = flask.make_response(data)
    resp.headers["Access-Control-Allow-Origin"]= "*"
    resp.headers["Access-Control-Allow-Methods"]= "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"]= '*'
    resp.headers["Status"]= "200 OK"
    resp.headers["Vary"]= "Accept"
    resp.headers['Content-Type']= 'image/png'

    return resp

@app.route("/api/v1/project/<int:project_id>/load")
def load_project(project_id):
    loc = load_localizer(project_id)
    if loc is None:
        return flask.make_response("Project not found", 404)
    localizers[project_id] = loc
    intrinsics[project_id] = loc.camera_matrix
    print(f'Loaded project {project_id}')    
    return flask.make_response("Project loaded successfully")

@app.route("/api/v1/project/<int:project_id>/intrinsics", methods=["POST"])
def add_intrinsics(project_id):
    if flask.request.method == "POST":
        if "camera_matrix" in flask.request.json:
            intrinsics[project_id] = np.array(flask.request.json["camera_matrix"])    
            return flask.make_response("Query camera intrinsics added successfully")  
        else:
            return flask.make_response("Query camera intrinsics not found", 404)
    else:
        return flask.make_response("Invalid request", 404)

@app.route("/api/v1/project/<int:project_id>/localize", methods=["POST"])
def localize_request(project_id):
    if project_id not in localizers:
        return flask.make_response("Project not loaded", 404)
    
    if not flask.request.method == "POST":
        return flask.make_response("Invalid request", 404)

    if flask.request.files.get("camera_matrix"):
        json_str=flask.request.files["camera_matrix"].read()
        camera_matrix = np.frombuffer(json_str, dtype="float").reshape(3,3)
    else:
        camera_matrix = intrinsics[project_id] # default to project intrinsics

    if flask.request.files.get("image"):
        img = Image.open(io.BytesIO(flask.request.files["image"].read()))   
    elif flask.request.data:
        img = Image.open(io.BytesIO(flask.request.data))   
    else:
        return flask.make_response("Image not found", 404)
     
    img = np.array(img)        

    loc = localizers[project_id]

    loc.query_img = img

    if loc.still_running:
        res = {'success':False}
        return flask.make_response(res)
    
    loc.still_running = True
    
    T_m1_c2, inliers, ret_idx = loc.callback_query(img, camera_matrix)
    if T_m1_c2 is None:
        res = {'success':False}
    else:
        pose = matrix2pose(T_m1_c2)
        loc.query_pose = pose
        res = {'success':True, 'pose':tuple(pose.tolist()), 'inliers':inliers, 'ret_imgs':tuple(ret_idx)}
        loc.ret_img = loc.load_rgb(ret_idx[0]) 
        loc.ret_pose = loc.get_pose(ret_idx[0])

    loc.still_running = False

    return flask.make_response(res)
        

@app.route("/api/v1/project/<int:project_id>/localize_multiple", methods=["POST"])
def localize_multiple_request(project_id):
    if project_id not in localizers:
        return flask.make_response("Project not loaded", 404)
    
    if not flask.request.method == "POST":
        return flask.make_response("Invalid request", 404)

    N_imgs = len(flask.request.files) - 2
    if N_imgs < 1:
        return flask.make_response("Incorrect number of inputs", 404)
    
    if flask.request.files.get("camera_matrix"):
        json_str=flask.request.files["camera_matrix"].read()
        camera_matrix = np.frombuffer(json_str, dtype="float").reshape(3,3)
    else:
        return flask.make_response("Intrinsics not found", 404)        

    if flask.request.files.get("poses"):
        json_str=flask.request.files["poses"].read()
        poses_l = np.frombuffer(json_str, dtype="float").reshape(-1,7)
        poses_l = poses_l.tolist()
    else:
        return flask.make_response("Poses not found", 404)  

    loc = localizers[project_id]
    I2_l = []
    
    for fkey in flask.request.files.keys():
        if "image" in fkey:            
            img = Image.open(io.BytesIO(flask.request.files[fkey].read()))    
            img = np.array(img)
            I2_l.append(img)

    T_m1_m2, inliers = loc.callback_query_multiple(I2_l, poses_l, camera_matrix)
    if T_m1_m2 is None:
        res = {'success':False}
    else:
        pose = matrix2pose(T_m1_m2)
        res = {'pose':tuple(pose.tolist()), 'inliers':inliers, 'success':True}

    return flask.make_response(res)



# post request method for uploading data to local filesystem for development
@app.route("/api/v1/project/<int:project_id>/upload", methods=["POST"])
def upload(project_id):
    if flask.request.method == "POST":
        if len(flask.request.data) > 0:

            with open(os.path.join("/Data.zip"), "wb") as f:
                f.write(flask.request.data)

            home = os.environ.get('HOME')
            sample_data = os.path.join(home, "datasets", f"project_{project_id}")
            os.makedirs(sample_data, exist_ok=True)

            thread = Thread(
                target=preprocess_task, args=(sample_data,project_id,)
            )
            thread.start()

            return "success"
        else:
            return flask.make_response("Data not found", 404)
    else:
        return flask.make_response("Invalid request", 404)

def preprocess_task(sample_data,project_id):
    print("started preprocessing...")
    shutil.rmtree(os.path.join("/Data"), ignore_errors=True)
    with zipfile.ZipFile("/Data.zip", "r") as zip_ref:
        zip_ref.extractall("/Data")
    shutil.rmtree(sample_data, ignore_errors=True)

    # remove output directory folders if they exist
    frame_rate = 2
    max_depth = 5
    voxel = 0.01
    # create preprocessor object
    home = os.environ.get('HOME')
    process = subprocess.Popen(["python3", "preprocessor/cli.py",
                                "-i", "/Data", "-o", sample_data, "-f", str(frame_rate), "-d", str(max_depth), "-v", str(voxel),
                                "--mobile_inspector"])
    process.wait()

    print('Reloading project...')  
    loc_1 = load_localizer(project_id) 
    loc_1.build_database()
    localizers[project_id] = loc_1
    intrinsics[project_id] = loc_1.camera_matrix     
    print(f'Loaded project {project_id}')    

if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=5000)
    init_ip_address()
    app.run(host='::',port=5000, debug=True)    
