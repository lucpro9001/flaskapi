from flask import Flask, request, Response
from flask import send_from_directory
import joblib
import os
import json
import pandas as pd
import datetime;
import numpy as np

class ModelCombiner:
    """
    Combine predictions of a list of fitted classification models by taking the average of their predicted probabilities.

    Parameters:
    models (list): A list of fitted Scikit-learn classification models.
    """
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        """
        Combine predictions of the fitted models by taking the average of their predicted probabilities.

        Parameters:
        X (array-like): The input data.

        Returns:
        A 1D array of predicted labels.
        """
        probs = np.array([model.predict_proba(X) for model in self.models])
        combined_probs = np.mean(probs, axis=0)
        combined_preds = np.argmax(combined_probs, axis=1)
        return combined_preds

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory("static", "index.html")

@app.route('/payload')
def categories():
    try:
        cate = {
            "Các biến chứng": [
                "Atypical angina",
                "Non-anginal pain",
                "Asymptomatic"
            ],
            "Điện tâm đồ": [
                0,
                1,
                2
            ],
            "Thể dục": [
                "Us",
                "U",
                "D",
                "Upsloping",
                "Ds",
                "Flat",
                "Downsloping",
                "F"
            ],
            "Mạch": [
                0,
                1,
                2,
                3
            ],
            "Thalassemia": [
                3,
                6,
                7
            ]
        }
        json_data = json.dumps(cate, ensure_ascii=False).encode('utf-16')
        return Response(json_data, mimetype='application/json; charset=utf-16')
    except:
        return 'internal error server', 500

@app.route('/api/predict/csv')
def predict_csv():
    try:
        if 'file' not in request.files:
            return 'No file uploaded', 400
        # use ordinal dummy encoding
        file = request.files['file']
        df = pd.read_csv(file.stream, encoding='utf-16', index_col=False)
        # validate file
        is_valid = validate_csv(df, False)
        if is_valid != 'ok':
            return is_valid, 400
        df['Giới tính'] = df['Giới tính'].map({'Female': 1, 'Male': 0}).astype(int)
        df['Đường huyết'] = df['Đường huyết'].astype(int)
        df['Đau ngực'] = df['Đau ngực'].map({'Yes': 1, 'No': 0}).astype(int)
        features = df.drop(columns=['id'], axis=1)
        features = pd.get_dummies(features, drop_first=True)
        columns = ['Tuổi', 'Huyết áp', 'Cholesterol', 'Nhịp tim', 'Trầm cảm']
        data_path = os.path.join(app.root_path, 'settings/ranges_data.pkl')
        ranges_ = joblib.load(data_path)
        for i in range(len(columns)):
            features[columns[i]] = features[columns[i]].apply(
                lambda x: ([i1 for i1, value in enumerate(([e[0] <= x <= e[1] for e in ranges_[i]])) if value])[0])
        # predict
        folder_path = os.path.join(app.root_path, 'models')
        models = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                models.append(joblib.load(file_path))  

        model_combiner = ModelCombiner(models)
        y_pred = model_combiner.predict(features)
        df['predicted'] = y_pred
        return df[['id', 'predicted']].to_csv(encoding='utf-16', index=False), 200
    except Exception as e:
        return str(e.args), 500

@app.route('/api/predict/csv', methods=['POST'])
def stored_csv():
    try:
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        df = pd.read_csv(file.stream, encoding='utf-16', index_col=False)
        # validate file
        is_valid = validate_csv(df, True)
        if is_valid != 'ok':
            return is_valid, 400
        date = (datetime.datetime.now()).strftime("%d-%m-%Y-%H-%M-%M")
        path = os.path.join(app.root_path, 'stored_csv', f'file_{date}.csv')
        df.to_csv(path, encoding='utf-16', index=False)
        return 'Created', 301
    except Exception as e:
        return str(e.args), 500

def validate_csv(df, have_target):
    names = ['id','Tuổi','Giới tính','Các biến chứng','Huyết áp','Cholesterol','Đường huyết','Điện tâm đồ','Nhịp tim','Đau ngực','Trầm cảm','Thể dục','Mạch','Thalassemia']
    valids = []
    if(have_target):
        names.append('Bệnh tim')
        names.remove('id')
    else:
        valids.append(True)
    cols = set(names)
    # valid fields
    if not cols.issubset(df.columns):
        return 'header false, pls check your csv header'
    
    valids.append((df['Tuổi'].between(1, 100) & (df['Tuổi'].astype(int) == df['Tuổi'])).any())
    valids.append((df['Giới tính'].isin(['Male', 'Female'])).any())
    valids.append((df['Các biến chứng'].isin(['Atypical angina', 'Non-anginal pain', 'Asymptomatic'])).any())
    valids.append((df['Huyết áp'].between(80, 200)).any())
    valids.append((df['Cholesterol'].between(120, 600)).any())
    valids.append((df['Đường huyết'].isin([True, False])).any())
    valids.append((df['Điện tâm đồ'].isin([0, 1, 2])).any())
    valids.append((df['Nhịp tim'].between(60, 220)).any())
    valids.append((df['Đau ngực'].isin(['Yes', 'No'])).any())
    valids.append((df['Trầm cảm'] >= 0).any())
    valids.append((
        df['Thể dục'].isin([
            "Us",
            "U",
            "D",
            "Upsloping",
            "Ds",
            "Flat",
            "Downsloping",
            "F"
        ])
    ).any())
    valids.append((df['Mạch'].isin([0,1,2,3])).any())
    valids.append((df['Thalassemia'].isin([3, 6, 7])).any())
    if have_target:
        valids.append((df['Bệnh tim'].isin([0,1,2,3,4])).any())
    if not any(valids):
        false_index = [index for index, value in enumerate(valids) if not value]
        return f"column {[names[i] for i in false_index if not i]} contains wrong value, pls check again"
    
    return 'ok'
if __name__ == '__main__':
    app.run()
