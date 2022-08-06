from Functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

header = ["Age", "Sex", "Chest_Pain", "Resting_BP", "Serum_Cholestrol/(mg/dl)", 
          "Fasting_Blood_Sugar (> 120 mg/dl)", "Resting_ECG", "Max_HR", "Exercise_Induced_Angina", "Oldpeak", "Slope",
         "CA", "thal", "Target"]

data = input("Enter the filename of the dataset: ")

def app():

    # Import data
    df = load_data(data, header)

    # Clean data
    df.replace("?", np.nan, inplace = True)
    df.dropna(subset = ["Target"], axis = 0, inplace = True)
    df.reset_index(drop = True, inplace = True)

    # Export clean data
    df.to_csv("clean_df.csv")
    X, y = create_target_and_predictors(data=df)
    Ans1 = input("Would you like to show a correlation heatmap? (Y/N)\n")
    if Ans1.upper() == "Y":

        heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
        # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
        plt.show()

    Ans2 = input("Would you like to show regression model results? (Y/N)\n")
    if Ans2.upper() == "Y":
        train_algorithm_with_cross_validation(10, 0.75, X=X, y=y)

    Ans3 = input("Would you like to train the SVM model? (Y/N)\n")
    if Ans3.upper() == "Y":
        SVM_Model(X,y)
    time.sleep(5)


app()


