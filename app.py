from flask import Flask
import pickle
import pandas as pd

from flask import Flask,request,render_template

app = Flask(__name__,template_folder='template')

model = pickle.load(open('modeldone.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    get  = pd.Series(request.form.values())
    prediction = model.predict(get)    
    # print('Output 0  =  Not spam(ham)\n Output 1 = Spam')
    return render_template('index.html', prediction_text='Predicted Output: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)