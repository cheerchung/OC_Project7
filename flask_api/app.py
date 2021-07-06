import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from os.path import join, dirname
import pandas as pd

app = Flask(__name__, template_folder='./Templates')
# set up data
data = pickle.load(open(join(dirname(__file__), 'models/data_heroku.pkl'), 'rb'))
test = data['test'].sort_index()
model = data['model']
imp = data['imputer']
# threshold of risk credit scoring: (note > threshold) -> high risk
threshold = 0.384

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["post"])
def predict():
    idx = request.form.get('id')
    gender = request.form.get('gender')
    amt = request.form.get('amt')
    age = request.form.get('age')
    edu = request.form.get('education')
    ext = request.form.get('ext3')

    # if client don't have an ID
    if not idx:
        client = pd.Series(index=test.columns, dtype=float)
        client['CODE_GENDER_F'] = int(gender)
        client['AMT_CREDIT'] = int(amt)
        client['DAYS_BIRTH'] = int(age)*(-365)
        client['NAME_EDUCATION_TYPE_Higher education'] = int(edu)
        client['EXT_SOURCE_3'] = float(ext)
        client_imp = imp.transform(client.values.reshape(1, -1))
    # if client has an ID
    else:
        client = test[test.index == int(idx)]
        client_imp = imp.transform(client)

    result = model.predict_proba(client_imp)[0, 1]
    if result > threshold:
        conclusion = 'high risk'
    elif result > threshold * 0.9:
        conclusion = 'medium risk'
    else:
        conclusion = 'low risk'
    output = np.round(result, decimals=2)
    output = str(output)
    return render_template('index.html',
                           prediction_text='client score should be {}, presenting {}'.format(output,
                                                                                             conclusion))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)

    idx = data['id']
    gender = data['gender']
    amt = data['amt']
    age = data['age']
    edu = data['education']
    ext = data['ext3']

    # if client don't have an ID
    if not idx:
        client = pd.Series(index=test.columns, dtype=float)
        client['CODE_GENDER_F'] = int(gender)
        client['AMT_CREDIT'] = int(amt)
        client['DAYS_BIRTH'] = int(age) * (-365)
        client['NAME_EDUCATION_TYPE_Higher education'] = int(edu)
        client['EXT_SOURCE_3'] = float(ext)
        client_imp = imp.transform(client.values.reshape(1, -1))
    # if client has an ID
    else:
        client = test[test.index == int(idx)]
        client_imp = imp.transform(client)

    output = model.predict_proba(client_imp)[0, 1]
    output = str(output)

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)