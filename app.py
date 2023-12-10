from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
from flask_bootstrap import Bootstrap



app = Flask(__name__,static_url_path='/static', static_folder='static')


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    from src.pipeline.predict_pipeline import CustomData,PredictPipeline
    if request.method == 'GET':
        return render_template('home.html')
    else:
        
        data = CustomData(
            gender=request.form.get('gender'),
            geographic_region=request.form.get('geographic_region'),
            loyalty_status=request.form.get('loyalty_status'),
            neighborhood_cluster=request.form.get('neighborhood_cluster'),
            affluence_grade=int(request.form.get('affluence_grade')),
            age=int(request.form.get('age')),
            loyalty_card_tenure=int(request.form.get('loyalty_card_tenure'))
        )

        pred_df = data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        prediction_result = results[0] 
        

        return render_template('result.html', prediction=prediction_result)


# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run(debug=True)
