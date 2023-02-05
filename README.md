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
https://drive.google.com/drive/u/0/folders/1bkKgJ0jFC7D7luyDrVyKB-NgfEb72lt2. The filename is Nat2021us.zip. Please note that this zip file is 222MB but unzipping it will generate a 4GB file. 

## childbirth_get_small_sample.ipynb
For performance reasons, we wrote a program called childbirth_get_small_sample.ipynb, randomly selecting a smaller sample from the massive 4GB file. We have pre-created two smaller data files. Nat2021us_small_30000.txt contains 30K rows, and Nat2021us_small_200000.txt contains 200K rows. Due to their size, we store only the compressed zip file in GitHub. Please unzip them to a directory called Nat2021us for the Jupyter notebook to load.

## childbirth_EDA.ipynb
childbirth_EDA parses the Nat2021us.txt, a fixed-length flat file, and creates a Pandas data frame. It then splits it into three groups - training, validation, and testing. The training data (60%) is for training the model. 
The validation data (20%) is for checking the trained model. Finally, the testing data (20%) is set aside for final comparison. 

We examined the document and selected 79 columns or features from the Nat2021us_small_200000.txt. Please reference the CDC data dictionary - "CDC ChildBirth Data UserGuide2021.pdf" to add more features. Starting page 8 of the document specifies each field's start and end position. Then modify childbirth_EDA.ipynb and add a line to the Data Extract cell. 

We use a convenient library called pandas_profiling, which summarizes each column's data distribution and saves it to a file called "train_profile.html." To install the pandas_profiling, please run "!pip install pandas_profiling". 




