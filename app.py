import os
import joblib
classifier = joblib.load(r'adult_autism.pkl')

# importing Flask and other modules
from flask import Flask, request, render_template 
import numpy as np
## import pandas as pd
  
# Flask constructor
app = Flask(__name__)   
  
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods = ["GET", "POST"])
def adult_autism():
    if request.method == "POST":
       A_Score = request.form.get("A_Score")
       age = request.form.get("age")
       p = classifier.predict(pd.array([A_Score, age]).reshape(-2,2))
       if p == 0:
           output = "Adult autism is predicted"
           return render_template('index.html', output=output)
       elif p == 1:
           output = "You are safe from adult autism"
           return render_template('index.html', output=output)
        
    return render_template('index.html')
  
if __name__=='__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port)
