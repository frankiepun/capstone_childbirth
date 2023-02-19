from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from childbirth_data_dictionary import *
import json
from childbirth_common_util import *
import numpy as np
import pandas as pd
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import math

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

column_list = {}
column_list['age'] = util_load_x_columns_list_from_file("age")
column_list['weight'] = util_load_x_columns_list_from_file("weight")
models = {}
models['age'] = util_load_models_from_file("age")
models['weight'] = util_load_models_from_file("weight")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("predict_input.html", {"request": request})


@app.get("/input", response_class=HTMLResponse)
def data_input(request: Request):
    default_value_dict =  dict(request.query_params)
    return templates.TemplateResponse("predict_input.html", {"request": request, "default_value_dict": default_value_dict})


# return the data dictionary for each feature
@app.get("/features/{predict_output_type}")
def get_all_features_data_dictionary(predict_output_type):

    if predict_output_type not in column_list:
        return childbirth_data_dict # default the return_dict in case output_type is not found

    column_list_by_type = column_list[predict_output_type]

    return_dict = {}
    for column in column_list_by_type:
        return_dict[column] = childbirth_data_dict.get(column, "ERROR - key not found in childbirth_data_dict.")

    return return_dict

@app.get("/help", response_class=HTMLResponse)
def get_help_and_example_payload(request: Request):
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

    return templates.TemplateResponse("help.html", {"request": request,
                                                    "get_age_url": f"{url_first_part}/age?{url_query_string_dict['age'][1:]}",
                                                    "post_age_url": f"{url_first_part}/age",
                                                    "get_weight_url": f"{url_first_part}/weight?{url_query_string_dict['weight'][1:]}",
                                                    "post_weight_url": f"{url_first_part}/weight",
                                                    "age_body_example": f"{json.dumps(input_array_age)}",
                                                    "weight_body_example": f"{json.dumps(input_array_weight)}"
                                                    })


@app.get("/predict")
def predict_http_form(request: Request, predict_output_type : str):
    dictobj =  predict_single_input(predict_output_type, request)
    input_url = request.url._url.replace("/predict?", "/input?")

    input_params_dict = dict(request.query_params)
    input_params_dict.pop('predict_output_type')
    input_translated_text_dict = {}
    for key, value in input_params_dict.items():
        input_translated_text_dict[key] = childbirth_data_dict[key][value]

    result = list(dictobj.values())[0]
    # the following is applicable for weight only.
    result_in_oz = result * 0.035274
    result_in_lb = math.floor(result_in_oz / 16)
    result_in_lb_oz = round(result_in_oz % 16)
    return templates.TemplateResponse("predict_result.html", {"request": request,
                                                              "predict_output_type": predict_output_type,
                                                              "result": result,
                                                              "result_in_oz": result_in_oz,
                                                              "result_in_lb": result_in_lb,
                                                              "result_in_lb_oz": result_in_lb_oz,
                                                              "input_translated_text_dict": input_translated_text_dict,
                                                              "complete_url": request.url._url,
                                                              "input_url": input_url,
                                                              "childbirth_data_display_text_dict": childbirth_data_display_text_dict
                                                              })





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


def main():
    print("Welcome to Childbirth Predictor")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()