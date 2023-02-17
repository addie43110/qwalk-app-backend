from flask import Flask, request, jsonify
import json
import os, glob

from helpers import get_response_image
from quantum_functions import create_plots

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@cross_origin
@app.route('/api/get_qw_multiple', methods=['POST'])
def get_qw_multiple():
    for f in glob.glob("./images/*.png"):
        os.remove(f)

    data = json.loads(request.data)
    
    dim = data['dimensions']
    num_states = data['num_states']
    iterations = data['iterations']

    create_plots(dim, num_states, iterations)

    encoded_imgs = {}
    for i in range(iterations+1):
        encoded_imgs[str(i)] = get_response_image('./images/dist'+str(i)+'.png')
    return jsonify(encoded_imgs)

if __name__ == "__main__":
    app.run(port=8000, debug=True)