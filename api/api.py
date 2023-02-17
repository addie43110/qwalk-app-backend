import time
from flask import Flask, request, render_template, send_file, jsonify, Response
import io
import json
import sys, os, glob

import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from helpers import get_response_image
from quantum_functions import qwalk, create_plots

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/time', methods={'GET'})
def get_current_time():
    return {'time': time.time()}

@app.route('/api/create_graphs', methods=['POST'])
def create_graphs():
    data = request.get_json()
    data['cat'] = 'none'
    return data

@app.route('/api/get_graph_test', methods=['GET'])
def get_graph_test():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@cross_origin()
@app.route('/api/get_qw_test', methods=['GET'])
def get_qw_test():
    dim = 3
    num_states = 64
    iterations = 2
    create_plots(dim, num_states, iterations)

    return send_file('images/dist1.png', mimetype='image/gif')
    #return render_template('untitled1.html', name = 'new_plot', url ='./images/new_plot.png')

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