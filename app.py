import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    df = pd.DataFrame(features_value, columns=['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership'])
    output = model.predict(df)
    result = np.round(output[0],2)
    return render_template('predict.html', prediction_text='₹{}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
