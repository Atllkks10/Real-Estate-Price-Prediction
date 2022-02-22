from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

data = pd.read_csv(r'C:\Users\atiak\Desktop\hepsi_emlak\product_data\processed_data.csv')
regression = load(open(r"C:\Users\atiak\Desktop\hepsi_emlak\models\gradient_tree_model.joblib", 'rb'))

def encode_neighborhood(neighborhood: str):
    A = []
    for i in data.columns[-21:]:
        if neighborhood.upper() in i.upper():
            A = A + [1]
        else:
            A = A + [0]
    return A

def predict_price(floor_area: int, age: int, floor: int, room_num : int , hall_num: int, neighborhood: str):
    A = [floor_area, age, floor,room_num, hall_num] + encode_neighborhood(neighborhood)
    A = np.array(A).reshape(1,-1)
    prediction = regression.predict(A)
    return prediction[0]

@app.route("/", methods = ['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', pred_val = 'Prediction')
    if request.method =='POST':
        floor_area = request.form['floor_area']
        age = request.form['age']
        floor = request.form['floor']
        room_num = request.form['room_num']
        hall_num = request.form['hall_num']
        neighborhood = request.form['neighborhood']
        return render_template('index.html', pred_val = str(predict_price(floor_area, age, floor, room_num, hall_num, neighborhood)))




