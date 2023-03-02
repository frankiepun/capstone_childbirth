# a file storing common utility functions for prediction weight and age

from datetime import datetime

import pickle
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from pandas_profiling import ProfileReport

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV

import tensorflow as tf

# model_path = "/content/drive/MyDrive/w210-capstone/Colab/models"
# local jupyer notebook
model_path="models"

# create an empty dataframe with the correct column name and type
def util_create_empty_X_input_df(predict_output_type):
    numeric_column_list = np.loadtxt(f'{model_path}/numeric_column_list.txt', dtype="object")
    column_list = np.loadtxt(f'{model_path}/feature_list_{predict_output_type}.txt', dtype="object")
    column_dict = {}
    for column in column_list:
        if column in ['combined_gestation_week', 'birth_weight_in_g']:
            continue
        if column in numeric_column_list:
            column_dict[column] = pd.Series(dtype='float')
        else:
            column_dict[column] = pd.Series(dtype='str')

    df = pd.DataFrame(column_dict)
    return df


# convert the column to proper type. only str(object) or float are used because they support NA
def util_change_column_type(X_features):
    numeric_column_list = np.loadtxt(f'{model_path}/numeric_column_list.txt', dtype="object")
    for column in X_features.columns:
        if column in numeric_column_list:
            X_features[column] = X_features[column].astype(float)
        else:
            X_features[column] = X_features[column].astype(str)
    return X_features


# replace unknown value such as 99 or 999 with feature mean
# unknown_value can be 9, 99 or 999 dependingo on the column
def util_replace_unknown_99_with_mean(X_features, column_name, unknown_value, predict_output_type):
    feature_mean_value_df = pd.read_csv(f"{model_path}/train_mean_{predict_output_type}.csv")

    if column_name in X_features.columns:
        X_features[column_name] = X_features[column_name].replace(unknown_value, np.nan)
        mean_value = feature_mean_value_df[column_name].item()
        X_features[column_name] = X_features[column_name].fillna(mean_value)
    return X_features


# always calculate BMI
def util_calc_bmi(X_features):
    util_replace_unknown_99_with_mean(X_features, 'prepregnancy_weight', 999, 'weight')
    util_replace_unknown_99_with_mean(X_features, 'mother_height_in_total_inches', 99, 'weight')
    X_features['bmi'] = (X_features['prepregnancy_weight'] / X_features['mother_height_in_total_inches'] / X_features['mother_height_in_total_inches']) * 703
    return X_features


# this function handles the NA and convert the column to proper types
# HTTP API accepts only string so the model must match the datatype
def util_handle_na_and_type(X_features, predict_output_type):

    feature_default_value_df = pd.read_csv(f"{model_path}/train_mode_{predict_output_type}.csv")

    # if X_features has fewer columns than the scaler_features, we must add the column back and default them to mode
    for feature in feature_default_value_df.columns:
        if feature not in X_features.columns:
            X_features[feature] = feature_default_value_df[feature].item()

    # convert the input feature to the correct type
    X_features = util_change_column_type(X_features)

    # calculate BMI from weight and height
    util_calc_bmi(X_features)

    # handle NA
    X_features['marital_status'] = X_features.marital_status.fillna(3)
    X_features['paternity_acknowledged'] = X_features.paternity_acknowledged.fillna('U')

    # for other categorical variables that have NA, fill it with the mode from the file
    for feature in X_features.columns:
        if feature in ['combined_gestation_week', 'birth_weight_in_g']:
            continue
        if X_features[feature].dtype == 'O':
            # replace it with mode
            mode_value = feature_default_value_df[feature].item()
            X_features[feature] = X_features[feature].replace("nan", mode_value)
        if X_features[feature].dtype != 'O': # flaot
            # for numeric columns, map unknown to mean
            if feature == "bmi":
                unknown_value = 99.9
            elif feature == "prepregnancy_weight":
                unknown_value = 999
            else:
                unknown_value = 99
            X_features = util_replace_unknown_99_with_mean(X_features, feature, unknown_value, predict_output_type)

    # convert the input feature to the correct type again
    X_features = util_change_column_type(X_features)

    return X_features


# At prediction phase, this function scales and transforms the values in X_input to mean = 0, std dev = 1
# It invokes util_handle_na() to handle NA cases
# X_input is the dataframe of features. It uses the saved pkl files to rank and scale the data
# X_input features must match the features in feature_rank_dict and standard_scalar files
# X_input features do not contain the output variables - birth_weight_in_g and combined_gestation_week
# and set the value to mean if it is missing
def util_scale(X_input, predict_output_type):
   
    with open(f'{model_path}/feature_rank_dict_{predict_output_type}.pkl', 'rb') as f:
        features_rank_dict = pickle.load(f)
    with open(f'{model_path}/standard_scaler_{predict_output_type}.pkl','rb') as f2:
        scaler = pickle.load(f2)

    X_input = X_input[scaler.feature_names_in_].copy()
    X_input = util_handle_na_and_type(X_input, predict_output_type)

    # it converts the values to a ranked number (see util_calc_save_scaler() for details)
    for feature in scaler.feature_names_in_:
        if X_input[feature].dtype == 'O':
            X_input[feature] = X_input[feature].map(features_rank_dict[feature])

    X_input = scaler.transform(X_input) # Fit scaler on and transform the data
    X_input = pd.DataFrame(X_input, columns=scaler.feature_names_in_) # Convert back into dataframe

    # if a feature does not carry a value, we will set it to the mean which is 0
    X_input.fillna(0, inplace=True)
      
    return X_input



# This function is invoked during the training phase. util_scale() is called during the prediction time
# X_features is a dataframe containing the features and output (weight and age)
# It uses the relationship between the feature and output to convert a category value to a numeric value
# This method is simpler than one-hot encoding because it reduces the number of columns
# predict_output_type is either "weight" or "age"
# It saves the feature_rank_dict and standard_scaler file. 
def util_calc_save_scaler(X_features, predict_output_type):
    # it turns out all of them can be treated as categorical variables. 
    X_features_rank = pd.DataFrame()
    features_rank_dict = {}
    
    if predict_output_type == 'weight':
        output_column = "birth_weight_in_g"
    else:
        output_column = "combined_gestation_week"

    # The following code ranks the value by its relation to birth_weight_in_g
    # Example: mother_nativity has 3 values, 1=born in the US, 2=outside, 3=unknown
    # instead of using these values for model, we want to rank them 0, 1, 2 by the mean(birth_weight_in_g)
    # let's say 
    #   mean(birth_weight_in_g) with value 1 (born in the US) is 3500 grams
    #   mean(birth_weight_in_g) with value 2 (outside) is 2500 grams
    #   mean(birth_weight_in_g) with value 3 (unknown) is 1500 grams
    #   then order them. In this case, we will assign 0 to 3 (unknown), 1 to 2 (outside), 2 to 1 (US)
    for feature in X_features.columns:
        if X_features[feature].dtype == 'O':
            labels_ordered = X_features.groupby([feature])[output_column].mean().sort_values().index
            # labels_ordered now contains the ranked order of each value
            # for example, it is Index(['N', 'Y', 'U', 'X'], N has highest birth_weight_in_g
            # convert labels_ordered to a dictionary {k:i}
            labels_ordered = {k:i for i,k in enumerate(labels_ordered, 0)}
            # labels_ordered is a dictionary {6:0, 12:1, ...}. It means if the value is 6, map it to 0
            # another example {'N': 0, 'Y': 1, 'U': 2, 'X': 3}. if the value is N, map it to 0
            features_rank_dict[feature] = labels_ordered
            # print(f"{feature}: {labels_ordered}")
            X_features_rank[feature] = X_features[feature].map(labels_ordered)
            #print(feature)
            #print(features_rank_dict[feature])
        else:
            X_features_rank[feature] = X_features[feature]

    with open(f'{model_path}/feature_rank_dict_{predict_output_type}.pkl', 'wb') as f:
        pickle.dump(features_rank_dict, f)
    
    # get all features except the output variables: birth_weight_in_g and combined_gestation_week
    feature_scale=[feature for feature in X_features_rank.columns if feature not in ['birth_weight_in_g', 'combined_gestation_week']]

    # write to file - the mode of each feature. They will be used as default
    X_features[feature_scale].mode().to_csv(f"{model_path}/train_mode_{predict_output_type}.csv", index=False)

    # all values will be standardized to standard normal distribution with mean = 0, std dev = 1
    scaler = StandardScaler() # Initialize StandardScaler object
    
    # save the fitted StandardScaler
    X_features_scaled = scaler.fit_transform(X_features_rank[feature_scale]) # Fit scaler on and transform the data
    with open(f'{model_path}/standard_scaler_{predict_output_type}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # output the mean and standard deviation. 
    train_mean = pd.DataFrame(data=scaler.mean_.reshape(1, len(scaler.mean_)), columns=scaler.feature_names_in_)
    train_mean.to_csv(f"{model_path}/train_mean_{predict_output_type}.csv", index=False)
    train_stddev = pd.DataFrame(data=scaler.mean_.reshape(1, len(scaler.mean_)), columns=scaler.feature_names_in_)
    train_mean.to_csv(f"{model_path}/train_stddev_{predict_output_type}.csv", index=False)
    
    X_features_transformed = pd.DataFrame(X_features_scaled, columns=feature_scale) # Convert back into dataframe

    return X_features_transformed
    
    
   
# baseline RMSE (root mean square error) of our outcome prediction for weight or age. 
# Our model must beat this baseline RMSE or it doesn't make sense to use the ML learning models.
# During EDA, we inspect the correlation between the features and the output variable. We discover only
# one feature that has high correlation with the weight. It is plurality (single, twins, triplet, etc). 
# However, since single birth is about 97%, it does not provide a baseline. As a result, we choose the 
# average as our baseline.
# RMSE (root mean square error) is used to calculate the accuracy of our model
# The mean of weight is 3249.16g and the baseline RMSE for weight is 588.13 
# The mean of age is 38.51 and the baseline RMSE for age is 2.51
# y_train is the output
# predict_output_type is "weight" or "age"
def util_calc_baseline(y_train, predict_output_type):
    print(f"the {predict_output_type}'s mean in training is {np.mean(y_train)}")

    y_mean = np.arange(y_train.shape[0])
    y_mean.fill(np.mean(y_train))

    # calculate mean square error
    mse = mean_squared_error(y_train, y_mean)

    rmse = np.sqrt(mse)
    print(f"{predict_output_type}: rmse={rmse}")



# It fits the X_train data to train the model, then it uses X_val to predict and compare against y_val to
# measure the RMSE (root mean square error). 
# It also saves the model to the file system - f"models/model_{model_name}_{predict_output_type}.sav"
# Please note that X_train and X_val must be scaled and transformed
# model_name - linear, gb, sgd, lgbm, etc
# predict_output_type - "weight" or "age"
# model is LinearRegression(), GradientBoostingRegressor(), SDGRegressor(), etc
def util_train_and_evaluate(model_name, predict_output_type, model, X_train_scaled, y_train, X_val_scaled, y_val):
    
    start_time = time.time()
    print(f"Start training model {model_name} for {predict_output_type} at {datetime.now()}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    model_filename = f"{model_path}/model_{model_name}_{predict_output_type}.sav"
    print(f"Saving {model_name} to file: {model_filename}")
    pickle.dump(model, open(model_filename, 'wb'))
    
    end_time = time.time()
    print(f"End time = {datetime.now()}, elapsed time = {end_time - start_time}")
    print(f"{model_name} for {predict_output_type}: rmse={rmse}")
    return model
    

# load column list from file but drop the output variables - combined_gestation_week and birth_weight_in_g
def util_load_x_columns_list_from_file(predict_output_type):
    
    col_from_files = np.loadtxt(f'{model_path}/feature_list_{predict_output_type}.txt', dtype="object")
    
    # remove the outcome variables
    col_features_only =  np.delete(col_from_files, np.where(col_from_files == 'combined_gestation_week'))
    col_features_only = np.delete(col_features_only, np.where(col_features_only == 'birth_weight_in_g'))
    
    return col_features_only

# load models from file    
def util_load_models_from_file(predict_output_type):
    models = {}
    for model in ['linear', 'gb', 'sgd', 'lgbm', 'xgb', 'rf']:
         models[model] = pickle.load(open(f'{model_path}/model_{model}_{predict_output_type}.sav', 'rb'))
    models['nn'] = tf.keras.models.load_model(f'{model_path}/model_nn_{predict_output_type}')

    return models


# This function takes X_input and predicts weight
# it scales X_input first, and then invoke a list of models for prediction. Finally, it uses the weighted average of each model's prediction
# X_input is a dataframe containing one or many rows for prediction
# column_list - only these columns will be used for prediction. The column_list must match the trained models. If None, it losds from the file system
# models are a list of trained models. If None, this function will load from file system
def util_ensemble_predict_weight(X_input, column_list, models):

    predict_output_type = "weight"

    model_proportion = {'linear':0.05, 'gb':0.25, 'sgd':0.05, 'lgbm':0.25, 'xgb':0.25, 'rf':0.05, 'nn':0.10}
    #model_proportion = {'linear':0.0, 'gb':1.0, 'sgd':0.0, 'lgbm':0.0, 'xgb':0.0, 'rf':0.0, 'nn':0.0}

    proportion_sum = 0.0 # sanity check
    predicted_result = np.zeros(len(X_input))

    for key, value in model_proportion.items():
        proportion_sum = proportion_sum + value
        print(f"predicting using {key} and its proportion is {value}")
        if key != 'nn':
            # print(models[key].predict(X_input))
            predicted_result = predicted_result + models[key].predict(X_input) * value
        else:
            nn_result = models['nn'].predict(X_input).squeeze()
            # print(nn_result)
            predicted_result = predicted_result + nn_result * value

    if proportion_sum != 1:
        print("Error: proportion sum should be 1")

    return predicted_result




# This function takes X_input and predicts age
# it scales X_input first, and then invoke a list of models for prediction. Finally, it uses the weighted average of each model's prediction
# X_input is a dataframe containing one or many rows for prediction
# column_list - only these columns will be used for prediction. The column_list must match the trained models. If None, it losds from the file system
# models are a list of trained models. If None, this function will load from file system
def util_ensemble_predict_age(X_input, column_list, models):

    predict_output_type = "age"

    # model_proportion = {'linear':0.05, 'gb':0.25, 'svr':0.05, 'sgd':0.05, 'lgbm':0.25, 'xgb':0.25, 'rf':0.05, 'nn':0.05}
    # model_proportion = {'linear':0.1, 'gb':0.25, 'sgd':0.05, 'lgbm':0.25, 'xgb':0.25, 'rf':0.1}
    model_proportion = {'linear':0.05, 'gb':0.25, 'sgd':0.05, 'lgbm':0.25, 'xgb':0.25, 'rf':0.05, 'nn':0.10}
    
    proportion_sum = 0.0 # sanity check
    predicted_result = np.zeros(len(X_input))

    for key, value in model_proportion.items():
        proportion_sum = proportion_sum + value
        print(f"predicting using {key} and its proportion is {value}")
        if key != 'nn':
            # print(models[key].predict(X_input))
            predicted_result = predicted_result + models[key].predict(X_input) * value
        else:
            nn_result = models['nn'].predict(X_input).squeeze()
            # print(nn_result)
            predicted_result = predicted_result + nn_result * value

    if proportion_sum != 1:
        print("Error: proportion sum should be 1")

    return predicted_result


