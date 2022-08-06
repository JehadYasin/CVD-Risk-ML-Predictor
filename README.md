# CVD-Risk-ML-Predictor

Analyzing a Heart Disease dataset. Predicting the risk of heart disease using multiple variables (risk factors), including:
Age : Age of the patient
Sex : Sex of the patient (1 = male; 0 = female)
exng : exercise-induced angina (1 = yes; 0 = no)
cp: Chest Pain type chest pain type
*   Value 1: typical angina
*   Value 2: atypical angina
*   Value 3: non-anginal pain
*   Value 4: asymptomatic
trtbps : resting blood pressure (in mm Hg)
chol : cholestoral in mg/dl fetched via BMI sensor
fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
rest\_ecg : resting electrocardiographic results
*   Value 0: normal
*   Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
*   Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach : maximum heart rate achieved 
slp: the slope of the peak exercise ST segment
*   Value 1: unsloping
*   Value 2: flat
*   Value 3: downsloping
caa: number of major vessels (0-3)
thall : thalassemia
*   Value 1: null
*   Value 2: fixed defect
*   Value 3: normal
*   Value 4: reversable defect

Files:
- heart.csv: main dataset
- clean_df.csv: DataFrame after cleaning

Modules:
- Functions.py: Includes all the functions used (e.g. loading data, ML algorithm).
- App.py: Am interactive command-line app for basic analysis of the dataset (heart.csv). This file uses functions from Functions.py.
- gui.py: A web-based GUI (gradio) to output the risk of developing CVD dependent on multiple inputs (values of risk factors). 

ML Algorithms:
- SVM (Support Vector Machine)
- Random Forest Regressor

Created by: Jehad Yasin
Date Published: 6th of August, 2022

