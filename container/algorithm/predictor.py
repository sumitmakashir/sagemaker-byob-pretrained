# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import flask

import pandas as pd
import numpy as np

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = pickle.load(open(os.path.join(model_path, 'model.pkl'), 'rb'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict_proba(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    """
    Input format: csv
    """
    # Convert from CSV to pandas
    #if flask.request.content_type == 'text/csv':
    #    data = flask.request.data.decode('utf-8')
    #    s = StringIO(data)
    #    data = pd.read_csv(s)
    
    """
    Input format: json. Takes input as a list with first element being list description and subsequent elements being dictionaries as individual records
    ['description', {'key1':'value1', 'key1':'value1','keyn':'valuen'}, {'key1':'value1', 'key1':'value1','keyn':'valuen'}]
    """
    if flask.request.content_type == 'application/json':
        print("Working with JSON input")
        s = flask.request.data.decode('utf-8')
        json_list = json.loads(s)
        json_list = json_list[1:]
        data = pd.DataFrame()
        for record in json_list:
            data = pd.concat([data, pd.DataFrame(record)], axis = 0, sort = True)
        for f in data.columns:
            if data[f].dtype == 'object':
                data = data.loc[data[f] != 'java.util.ArrayList']

        # Feature engineering code
        data = data
            
        # list of features your model requires 
        feature_names = []
        data = data[feature_names].copy()

    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    #print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)
    predictions = [p[1] for p in predictions]

    # Convert from numpy back to CSV
    out = StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
