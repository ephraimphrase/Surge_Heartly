# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_heart_attack_data():

    #checking for the csv file in same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'heart.csv')


    # Load heart attack dataset
    df = pd.read_csv(csv_path)
    
    # Split dataset into input features (X) and output labels (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y


def train_logistic_regression(X_train, y_train):
    # Initialize logistic regression model
    lr_model = LogisticRegression(max_iter=1000)

    # Train model on training set
    lr_model.fit(X_train, y_train)

    return lr_model


def train_random_forest(X_train, y_train):
    # Initialize random forest model
    rf_model = RandomForestClassifier()

    # Train model on training set
    rf_model.fit(X_train, y_train)

    return rf_model


def train_svm(X_train, y_train):
    # Initialize SVM model
    svm_model = SVC()

    # Train model on training set
    svm_model.fit(X_train, y_train)

    return svm_model


def evaluate_model(model, X_test, y_test):
    # Make predictions on testing set
    y_pred = model.predict(X_test)

    # Evaluate model accuracy

    cm = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f"Classification report for {model}")
    print(classification_report(y_test, y_pred))
    print(f'Confusion Matrix for {model}')
    print(confusion_matrix(y_test, y_pred))

    #plot roc_auc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Get the ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred)

    # plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Plot the ROC-AUC curve
    plt.plot(fpr, tpr, label=f'{model} (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return acc