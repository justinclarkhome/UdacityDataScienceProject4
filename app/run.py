import json
import plotly
import plotly.express as px
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
import plotly.graph_objs as go
import os

app = Flask(__name__)
USER_IMAGE_UPLOAD_DIR = './user_images'
if not os.path.exists(USER_IMAGE_UPLOAD_DIR):
    os.mkdir(USER_IMAGE_UPLOAD_DIR)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=[1, 2, 3, 4, 5],
                    y=[20, 30, 20, 10, 20],
                )
            ],
            'layout': {
                # 'title': 'The image uploaded is',
                # 'xaxis': {'tickangle': 45},
                # 'yaxis': {'title': 'Frequency of True Values in Category'},
            }
        },
   ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('Upload an image', '')

    # use model to predict classification for query

    # This will render the go.html Please see that file. 
    # return render_template(
    #     'go.html',
    #     query=query,
    #     classification_result=classification_results
    # )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()