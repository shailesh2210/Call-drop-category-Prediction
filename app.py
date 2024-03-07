from flask import Flask , render_template , request
import pickle
import numpy as np

model = pickle.load(open("model\\rf_model.pkl", "rb"))

app = Flask(__name__) 
  
@app.route('/') 
def welcome(): 
    return render_template("index.html")
  
@app.route("/predict")
def predict():
    operator = request.form["operator"]
    inout_travelling = request.form["inout_travelling"]
    network_type = request.form["network_type"]
    rating = request.form["rating"]
    latitude = request.form["latitude"]
    longitude = request.form["longitude"]
    state_name = request.form["state_name"]
    month = request.form["month"]

    prediction = model.predict([[operator,inout_travelling,network_type,rating,latitude,longitude,state_name,month]])

    print(np.round(prediction))

    return render_template("predict.html", prediction=prediction)
  
# Start with flask web app, with debug as True,# only if this is the starting page 
if __name__ == "__main__": 
    app.run(debug=True) 