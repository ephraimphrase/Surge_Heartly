import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the heart disease dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'heart.csv')

dataset = pd.read_csv(csv_path)

#Split into features and targets
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the input data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler object for later use
joblib.dump(scaler, 'heart_attack_scaler.pkl')

def train_neural_network(X_train, y_train):
    # Define and train the neural network model

    model = Sequential()

    model.add(Dense(units=16, activation='relu', input_dim=13))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

    # Save the trained model as an .h5 file
    model.save('heart_attack_neural_network_model.h5')

    # Load the trained neural network model
    model = load_model('heart_attack_neural_network_model.h5')

    return model


# Define a function to make predictions using the loaded neural network model
def predict(model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Convert the input values to a NumPy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input data using the saved scaler object
    scaler = joblib.load('heart_attack_scaler.pkl')
    input_data = scaler.transform(input_data)

    # Make a prediction using the neural network model
    result = model.predict(input_data)[0][0]

    # Round the result to 2 decimal places and convert to a percentage
    probability = round(result * 100, 2)

    # Return the result and probability as a tuple
    return (result, probability)

def evaluate_nn(model, X_test, y_test):
    #Make a prediction based on test sets
    y_pred = model.predict(X_test)
    
    y_pred_labels = [1 if pred >= 0.5 else 0 for pred in y_pred]

    cm = confusion_matrix(y_test, y_pred_labels)

    acc = accuracy_score(y_test, y_pred_labels)
    print("Classification report for neural network: ")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix for neural network: ")
    print(confusion_matrix(y_test, y_pred_labels))
    print(acc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_labels)

    # Get the ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_labels)

    # plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Plot the ROC-AUC curve
    plt.plot(fpr, tpr, label='Neural Network (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return acc
