import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Create target variable and predictor variables

def load_data(data, headers: list):

    df = pd.read_csv(data)
    df.columns = headers
    
    return df

def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "Target"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm
def train_algorithm_with_cross_validation(
    K,
    SPLIT,
    X: pd.DataFrame = None, 
    y: pd.Series = None,
):
    """
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each
    fold during training.

    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable

    :return
    """

    Ans = input("Do you want to show a graphical display? (Y/N)\n")

    # Create a list that will store the accuracies of each fold
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

    if Ans.upper() == "Y":
        features = [i.split("__")[0] for i in X.columns]
        importances = model.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 20))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

def SVM_Model(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 13, svd_solver="full")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Getting accuracy of SVM predictions
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n")
    for i in cm:
        print(cm)
    print("")
    print(f"Accuracy: {accuracy_score(y_test, y_pred).round(2)}\n")

log = []

def Dec_SVM_Model(a1):

    header = ["Age", "Sex", "Chest_Pain", "Resting_BP", "Serum_Cholestrol/(mg/dl)", 
          "Fasting_Blood_Sugar (> 120 mg/dl)", "Resting_ECG", "Max_HR", "Exercise_Induced_Angina", "Oldpeak", "Slope",
         "CA", "thal", "Target"]
    df = load_data("heart.csv", header)

    # Clean data
    df.replace("?", np.nan, inplace = True)
    df.dropna(subset = ["Target"], axis = 0, inplace = True)
    df.reset_index(drop = True, inplace = True)

    X, y = create_target_and_predictors(data=df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 1, svd_solver = "full")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X_train, y_train)
    data = a1.to_numpy()
    sc.fit(data)
    health_measures = sc.transform(data)
    pca.fit(health_measures)
    health_measures = pca.transform(health_measures)
    prediction = classifier.predict(health_measures)[0]
    log.append((data, prediction))
    print(prediction)

    return "> 50% diameter narrowing. Increased risk of heart attack" if prediction == 1 else "< 50% diameter narrowing. Decreased risk of heart attack"

