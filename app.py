import os

import json
import tensorflow as tf
from flask import Flask, render_template, request, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np


# Set TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# PATHWAY_DICT_PATH = 'my_data/precily_pathways_dict.npy'
PRECILY_CELL_LINE_DICT_PATH = 'my_data/celline_pathways_dict.npy'
PRECILY_DRUG_DICT_PATH = 'my_data/drug_pathways_dict.npy'
# GDSC_DICT_PATH = 'my_data/precily_labels_dict.npy'f
GDSC_DICT_PATH = 'my_data/gdsc1_gdsc2_dict.npy'
SMILES_PATH = 'my_data/canonical_smiles.json'
PACCMANN_PATH = 'my_data/paccmann_array.npy'
MODEL_PATH_prc = 'tf2_model_all/precily_cv_5.hdf5'  # Update with the actual path
MODEL_PATH_tcnn = 'tf2_model_all/best_model.h5'  # Update with the actual path


def calculate_y(z):
    """
    Calculate y from z using the inverse of the formula: y = (1/z - 1)**(-10)

    Parameters:
    z (float): Input value for z

    Returns:
    float: Calculated value for y
    """
    return (1/z - 1)**(-10)

def calculate_smiles(drug_name):
    # load the SMILES json file
    with open(SMILES_PATH, 'r') as fp:
        canonical_smiles = json.load(fp)

    canonical_smile = canonical_smiles[drug_name]
    return canonical_smile

def calculate_genetic_feature(cell_line_name):
    # load genetic feature dictionary
    cell_mut_dict = np.load("my_data/cell_mut_matrix.npy", allow_pickle=True, encoding="latin1").item()
    cell_mut = cell_mut_dict["cell_mut"]
    cell_dict = cell_mut_dict['cell_dict']

    # calculate id-number of a cell
    cell_id = cell_dict[cell_line_name]
    cell_genetic_feature = cell_mut[cell_id]
    return cell_genetic_feature

def preprocess_input_tcnn(drug_name, cell_line_name):
    # Assuming drug_data and cell_data are single elements, convert them to numpy arrays
    drug_data = np.array([calculate_smiles(drug_name)])
    drug_data = np.transpose(drug_data, (0, 2, 1))  # Transpose the last two dimensions

    cell_data = np.array([calculate_genetic_feature(cell_line_name)])

    # print(f'DRUG Name: {drug_data}, CELL data: {cell_data}')

    # Load the model
    model_tcnn = tf.keras.models.load_model(MODEL_PATH_tcnn)

    # Make predictions using the loaded model
    predictions = model_tcnn.predict([drug_data, cell_data])

    # Return the predictions
    return predictions[0][0]

def preprocess_precily_input(drug_name, cell_line_name):
    cell_line_pathways_dict = np.load(PRECILY_CELL_LINE_DICT_PATH, allow_pickle=True).item()
    drug_pathways_dict = np.load(PRECILY_DRUG_DICT_PATH, allow_pickle=True).item()
    
    precily_model = tf.keras.models.load_model(MODEL_PATH_prc)
    
    cell_line_name = str(cell_line_name)
    drug_name = str(drug_name)
    
    if cell_line_name not in cell_line_pathways_dict or drug_name not in drug_pathways_dict:
        return "error"
    else:
        cell_line_pathways = np.array(cell_line_pathways_dict[cell_line_name]).reshape(1,-1)
        drug_pathways = np.array(drug_pathways_dict[drug_name]).reshape(1,-1)
        pathway_info = np.concatenate((cell_line_pathways, drug_pathways), axis=1)
        predictions = precily_model.predict(pathway_info)
        return predictions[0][0]


def preprocess_input_gdsc(drug_name, cell_line_name):
    # Make predictions using the loaded model
    gdsc_dict = np.load(GDSC_DICT_PATH, allow_pickle=True).item()
    print(f'GDSC Dict: {len(gdsc_dict)}')

    # key = (str(cell_line_name), str(drug_name))
    key = (str(drug_name), str(cell_line_name))
    predictions = gdsc_dict.get(key,"error")
    return predictions

def preprocess_input_paccmann(drug_name, cell_line_name):
    paccmann_dict = np.load(PACCMANN_PATH, allow_pickle=True).item()
    print(f'PACCMANN Dict: {len(paccmann_dict)}')

    # Make predictions using the loaded model
    key = (str(drug_name),str(cell_line_name))
    predictions = paccmann_dict.get(key,"error")
    return predictions



# instance of a flask object
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get input values from the form
        drug_name = request.form['drug_name']
        cell_line_name = request.form['cell_line_name']

        # Preprocess the input data and make predictions
        prediction = preprocess_input_gdsc(drug_name, cell_line_name)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')


@app.route('/precily', methods=['POST', 'GET'])
def predict_precily():
    if request.method == "POST":
        query = request.get_json()
        drug_name = query['drug_name']
        cell_line_name = query['cell_line_name']
        prediction = preprocess_precily_input(drug_name, cell_line_name)
        results = {'prediction': str(round(prediction, 4))}
        return jsonify(results)

@app.route('/tcnn', methods=['POST', 'GET'])
def predict_tcnn_ic50():
    if request.method == "POST":
        query = request.get_json()
        drug_name = query['drug_name']
        cell_line_name = query['cell_line_name']
        raw_prediction = preprocess_input_tcnn(drug_name, cell_line_name)
        prediction = calculate_y(raw_prediction)
        results = {'prediction': round(prediction, 4)}
        return jsonify(results)

@app.route('/gdsc', methods=['POST', 'GET'])
def predict_gdsc():
    if request.method == "POST":
        query = request.get_json()
        drug_name = query['drug_name']
        cell_line_name = query['cell_line_name']
        prediction = preprocess_input_gdsc(drug_name, cell_line_name)
        results = {'prediction': str(round(prediction, 4))}
        return jsonify(results)


@app.route('/paccmann', methods=['POST', 'GET'])
def predict_paccmann():
    if request.method == "POST":
        query = request.get_json()
        drug_name = query['drug_name']
        cell_line_name = query['cell_line_name']
        prediction = preprocess_input_paccmann(drug_name, cell_line_name)
        results = {'prediction': str(round(prediction, 4))}
        return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=True, port=3000)
