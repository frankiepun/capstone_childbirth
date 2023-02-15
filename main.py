from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from childbirth_data_dictionary import *
import json
from childbirth_common_util import *
import numpy as np
import pandas as pd


app = FastAPI()
column_list = {}
column_list['age'] = util_load_x_columns_list_from_file("age")
column_list['weight'] = util_load_x_columns_list_from_file("weight")
models = {}
models['age'] = util_load_models_from_file("age")
models['weight'] = util_load_models_from_file("weight")


@app.get("")
def read_root():
    return {"Hello": "World"}


@app.get("/features/{predict_output_type}")
def get_all_age_features(predict_output_type):

    if predict_output_type not in column_list:
        return childbirth_data_dict # default the return_dict in case output_type is not found

    column_list_by_type = column_list[predict_output_type]

    return_dict = {}
    for column in column_list_by_type:
        return_dict[column] = childbirth_data_dict.get(column, "ERROR - key not found in childbirth_data_dict.")

    return return_dict

@app.get("/help", response_class=HTMLResponse)
def get_help():
    # single request
    url_first_part = "http://127.0.0.1:8000/predict"

    predict_type_dict = {'age': {}, 'weight': {}} # key=age/weight. value=dict of column (key=column name, value=dict of values).
    url_query_string_dict = {}  # the key is age or weight. the value is the query string
    input1_dict = {'age': {}, 'weight': {}}  # key=age/weight, value=dict of column and first value

    for predict_output_type in ['age', 'weight']:
        for column in column_list['age']:
            predict_type_dict[predict_output_type][column] = childbirth_data_dict.get(column, ["ERROR - key not found in childbirth_data_dict."])
        url_query_string_dict[predict_output_type] = ""
        for key, val in predict_type_dict[predict_output_type].items():
            url_query_string_dict[predict_output_type] = f"{url_query_string_dict[predict_output_type]}&{key}={next(iter(val))}"
            input1_dict[predict_output_type][key] = next(iter(val))

    # an array of requests
    input_array_weight = [input1_dict['weight'], input1_dict['weight'],input1_dict['weight']]
    input_array_age = [input1_dict['age'], input1_dict['age'], input1_dict['age']]

    # HTML output
    html_content = f"""
    <html>
        <head>
            <title>Example HTML</title>
        </head>
        <body>
            <h2>Predict age by a single input by HTTP Get</h2>
            <code>{url_first_part}/age?{url_query_string_dict['age'][1:]}</code>
            <h2>Predict age by multiple inputs by HTTP Post</h2>
            <code>http POST url: {url_first_part}/age</code>
            <p>
            <code>http POST Body:</code>
            <code>{json.dumps(input_array_age)}</code>
            <p>
            <h2>Predict weight by a single input by HTTP Get</h2>
            <code>{url_first_part}/weight?{url_query_string_dict['weight'][1:]}</code>
            <h2>Predict weight by multiple inputs by HTTP Post</h2>
            <code>http POST url: {url_first_part}/weight</code>
            <p>
            <code>http POST Body:</code>
            <code>{json.dumps(input_array_weight)}</code>
        </body>
    </html>
    
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/predict/{predict_output_type}")
def predict_single_input(predict_output_type, request: Request):
    params = dict(request.query_params)

    X_input_df = util_create_empty_X_input_df(predict_output_type)
    X_input_df = X_input_df.append([params], ignore_index=True)

    #print(X_input_df[['birth_month', 'mother_age', 'mother_race1', 'father_age', 'bmi']])
    X_scaled = util_scale(X_input_df, predict_output_type)
    #print(X_scaled[['birth_month', 'mother_age', 'mother_race1', 'father_age', 'bmi']])
    y_pred = util_ensemble_predict_age(X_scaled, column_list[predict_output_type], models[predict_output_type])
    print(y_pred)
    return {predict_output_type: y_pred.item()}


# this endpoint handle multiple inputs in JSON format in the request body.
@app.post("/predict/{predict_output_type}")
async def predict_multiple_input(predict_output_type, request: Request):
    body = await request.json()
    print(f"The number of records in input = {len(body)}")

    X_input_df = util_create_empty_X_input_df(predict_output_type)
    X_input_df = X_input_df.append(body, ignore_index=True)
    #print(X_input_df[['birth_month', 'mother_age', 'mother_race1', 'father_age', 'bmi']])

    X_scaled = util_scale(X_input_df, predict_output_type)
    #print(X_scaled[['birth_month', 'mother_age', 'mother_race1', 'father_age', 'bmi']])
    y_pred = util_ensemble_predict_age(X_scaled, column_list[predict_output_type], models[predict_output_type])
    print(y_pred)
    return dict(enumerate(y_pred))



