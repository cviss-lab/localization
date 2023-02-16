import logging
import os
import flask
import typing

import localization
from libs.utils.loader import *
from libs.utils.projection import *
import io

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

loc_1 = load_localizer(1)
localizers[1] = loc_1
intrinsics[1] = loc_1.camera_matrix

app.logger.info("API server ready")

@app.route("/api/v1/project/<int:project_id>/load")
def load_project(project_id):
    loc = load_localizer(project_id)
    if loc is None:
        return flask.make_response("Project not found", 404)
    localizers[project_id] = loc
    intrinsics[project_id] = loc.camera_matrix
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
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            loc = localizers[project_id]

            img = Image.open(io.BytesIO(flask.request.files["image"].read()))    
            img = np.array(img)

            camera_matrix = intrinsics[project_id]
            T_m1_c2, inliers = loc.callback_query(img, camera_matrix)
            if T_m1_c2 is None:
                res = {'success':False}
            else:
                pose = matrix2pose(T_m1_c2)
                res = {'pose':tuple(pose.tolist()), 'inliers':inliers, 'success':True}

            return flask.make_response(res)

        else:
            return flask.make_response("Image not found", 404)
    else:
        return flask.make_response("Invalid request", 404)


if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=5000)
    app.run(host='::',port=5000)    
