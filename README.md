# W210 Capstone Childbirth Prediction

### By Frankie Pun, Luis Delgado, Heather Rancic, Irene Shaffer

### Spring 2023

## Project Summary

The birth weight of an infant may come as a surprise at the end of a pregnancy. The birthweight is an important factor in the 
health of the mother and the delivery procedure of the infant. If the parents know the birthweight in advance, they may be 
more likely to schedule an early induction or opt for a Cesarean delivery to avoid complications that can result from larger 
babies, such as shoulder dystocia. In addition, a higher baby weight may allow the parents to skip purchasing certain baby 
items which are only needed for very small infants.

## Data Source
The full CDC 2021 childbirth data has been uploaded to: 
https://drive.google.com/drive/u/0/folders/1bkKgJ0jFC7D7luyDrVyKB-NgfEb72lt2. The filename is *Nat2021us.zip*. Please note that this zip file is 222MB but unzipping it will generate a 4GB file. 

## childbirth_get_small_sample.ipynb
For performance reasons, we wrote a program called *childbirth_get_small_sample.ipynb*, randomly selecting a smaller sample from the massive 4GB file. We have pre-created two smaller data files. *Nat2021us_small_30000.txt* contains 30K rows, and *Nat2021us_small_200000.txt* contains 200K rows. Due to their size, we store only the compressed zip file in GitHub. Please unzip them to a directory called *Nat2021us* for the Jupyter notebook to load.

## Colab
We decided to use Colab for our primary development environment because it provides a standardized and shared development environment. Please refer to the colab files in the shared google drive (https://drive.google.com/drive/folders/1jms-QHD3Lmsh1B0ZJ82mboKpvYgWO5dx?usp=share_link) for the latest version. We will regularly download and check in a copy to github.


## childbirth_EDA.ipynb

*childbirth_EDA.ipynb*  parses the *Nat2021us.txt*, a fixed-length flat file, and creates a Pandas dataframe. It then splits it into three groups - training, validation, and testing. The training data (60%) is for training the model. The validation data (20%) is for checking the trained model. Finally, the testing data (20%) is set aside for final comparison. 

We examined the document and selected 79 features from the *Nat2021us.txt*. Please reference the CDC data dictionary - *CDC ChildBirth Data UserGuide2021.pdf* for details about each feature. To add more features, please use the field's start and end position specified on page 8 of the document 

We use a convenient library called *pandas_profiling*, which summarizes each column's data distribution and saves it to a file called *train_profile.html*. To install the pandas_profiling, please run "!pip install pandas_profiling==3.6.3". 

*childbirth_EDA.ipynb* uses heatmaps to illustrate the correlation between features. For highly correlated features, only one feature will be selected. We will also drop highly skewed features.

The final result of EDA is a list of features to be used in training the models. A file called *models/feature_list.txt* is generated. It is a simple text file, and each line contains a feature name. We have considered multiple feature selection techniques to pick features, such as PCA or Lasso. Still, preliminary prediction results show that features confirmed by subject matter experts (OBGYN and Pediatricians) yield the best results. 

To run the EDA on colab, a cloud development environment capable of loading and processing the massive 4GB data file, please search for the keyword *colab* in the notebook and uncomment the code.

## childbirth_model_age.ipynb

This notebook uses supervised machine learning to predict the gestation age. It reads the train, validation, and test data files and the feature list from *models/feature_list_age.txt*. Feel free to edit *feature_list_age.txt" to add or remove features for the models. It uses several utility functions in a common python file *childbirth_common_util.py* to load the feature list and train the models. 

The baseline for the model prediction is the mean gestation age. During EDA, we correlated each feature against gestation age but didn't discover any feature with strong prediction power. As a result, we must resort to the most basic measurement - the average gestation age, which is 38.50 weeks. 

To measure the model's performance and accuracy, we choose RMSE (Root Mean Square Error), the most common performance indicator for a regression model. RMSE measures the average difference between values predicted by a model and the actual values. The RMSE for the baseline prediction, i.e., mean gestation age, is 2.51. 

Scaling is essential for transforming the features to a standard range of numeric values. Instead of one-hot encoding, we decided to use a ranking technique to convert a categorical feature to a numeric feature. For example, if a feature contains four values - Y, N, U, X, we first rank them based on their mean gestation_age. The lowest is assigned 0, and the second lowest is assigned 1. Please refer to the function *util_calc_save_scaler* in *childbirth_common_util.py* for the algorithm. Then we applied sklearn's StandardScaler to scale each feature to a standard range with mean = 0 and standard deviation = 1. We have also tried other scalers, such as MinMaxScaler and RobustScaler, but  StandardScaler yields the best results. 

For training the model, we applied the ensemble modeling technique, which combines multiple models to generate the optimal result. The base models are Linear Regression, Gradient Boosting Regressor, SGD Regressor, LGBM Regressor, Random Forest Regressor, and Neural Network. We have also tried other models, such as KNN and SVM, but they are dropped due to poor results. 

We tuned the base model by trying many combinations of hyperparameters. Please see this file *childbirth_model_parameter_tuning.ipynb* for details about the hyperparameter tuning. Please be aware that the tuning takes many hours to complete. 

The prediction is a weighted average of the result of the above six models. The weight of each model results from many trial-and-error, and we discover the current weights yield the optimal result without overfitting or underfitting. As a result, the RMSE of our model based on the test dataset is 2.02, better than our baseline of 2.51. 


## childbirth_model_weight.ipynb

The structure of *childbirth_model_weight.ipynb* is similar *childbirth_model_age.ipynb*. To add or remove features, please change the file *models/feature_list_weight.txt* and then run each cell to re-train the model and measure the result. The baseline is the average newborn's weight is 3249.15, and RMSE is 588.13. The model's RMSE is 472.52, better than the baseline's RMSE 588.13.

## Model Tuning
We can perform a few things to optimize the models:
### add or delete a feature. 
The features are stored in the files feature_list_age.txt or feature_list_weight.txt, under "/content/drive/MyDrive/w210-capstone/Colab/models". They are text files and each line contains a feature. 

### hyperparameter tuning
We are using six base linear regression models. Each model has a set of hyperparameters that can be tuned to improve the performance. Please refer to childbirth_model_parameter_tuning.ipynb for some example code that uses GridSearch to test different parameter combinations. 


### Add new model
Sklearn has many regression models (https://scikit-learn.org/stable/supervised_learning.html). So we can add more base models to our ensemble model to improve the performance.


## childbirth_model_parameter_tuning.ipynb

This notebook contains code to tune individual models. It uses Grid Search to identify the optimal hyperparameters for each model by trying different parameter combinations. Hyperparameters are the parameters that define and customize the base model. Please note that it takes a long time (many hours) for GridSearch to find the optimal parameters. 


## childbirth_common_util.py

This python file contains many common utility functions for predicting the gestation age or newborn weight. To use the model for prediction, a program must invoke util_load_x_columns_list_from_file() to load a list of input features and util_load_models_from_file() to load the trained models. Then it calls util_scale() to scale the input features to a standard range with mean=0 and standard deviation=1. Finally, it executes the util_ensemble_predict_weight() or util_ensemble_predict_age(), which accept the input features and predict the result.

This python file also provides common utility functions for training the model - util_handle_na() to handle NA values. util_calc_save_scaler() ranks and scales the input features and saves the scaler to the file system. util_calc_baseline() calculates the baseline and its root mean square error. util_train_evaluate() trains the individual base model and evaluates its accuracy. 
