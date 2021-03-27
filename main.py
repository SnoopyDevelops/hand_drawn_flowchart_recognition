import os

from flask import request, jsonify
from werkzeug.utils import secure_filename

from app import app
from flowchart_recognition import flowchart

ALLOWED_EXTENSIONS = {'jpg', 'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request', 'status': 400})
        resp.status_code = 400
        return resp

    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading', 'status': 400})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # path = secure_filename(file.filename)
        file.save(path)

        if '-p' in request.values.keys():
            padding = int(request.values['-p'])
        else:
            padding = 25

        if '-o' in request.values.keys():
            offset = int(request.values['-o'])
        else:
            offset = 10

        if '-a' in request.values.keys():
            arrow = int(request.values['-a'])
        else:
            arrow = 30

        nodes = flowchart(
            filename=path, padding=padding, offset=offset, arrow=arrow, gui=False
        )

        os.remove(path)
        resp = jsonify({'message': 'File successfully uploaded', 'data': nodes, 'status': 200})
        resp.status_code = 201
        return resp

    else:
        resp = jsonify({'message': 'Allowed file types are jpg and png', 'status': 400})
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
