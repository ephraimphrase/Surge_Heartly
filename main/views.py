from django.shortcuts import render
from .prediction_system import load_heart_attack_data, train_logistic_regression, train_random_forest, train_svm, evaluate_model
import numpy as np
from .neural_network import predict, train_neural_network, evaluate_nn

# Create your views here.

def heart_attack_prediction(request):
    if request.method == "POST":
        # Read input data from user
        age = int(request.POST.get("age"))
        sex = int(request.POST.get("sex"))
        cp = int(request.POST.get("cp"))
        trestbps = int(request.POST.get("trestbps"))
        chol = int(request.POST.get("chol"))
        fbs = int(request.POST.get("fbs"))
        restecg = int(request.POST.get("restecg"))
        thalach = int(request.POST.get("thalach"))
        exang = int(request.POST.get("exang"))
        oldpeak = float(request.POST.get("oldpeak"))
        slope = int(request.POST.get("slope"))
        ca = int(request.POST.get("ca"))
        thal = int(request.POST.get("thal"))

        # Load dataset
        X, y = load_heart_attack_data()

        # Train logistic regression model
        lr_model = train_logistic_regression(X, y)

        # Train random forest model
        rf_model = train_random_forest(X, y)

        # Train SVM model
        svm_model = train_svm(X, y)

        # Train Neural Network
        nn_model = train_neural_network(X, y)

        # Create input vector
        input_data = np.array(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make predictions using each model
        lr_pred = lr_model.predict(input_data)
        rf_pred = rf_model.predict(input_data)
        svm_pred = svm_model.predict(input_data)

        nn_pred, nn_poss = predict(nn_model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # Evaluate model accuracies
        lr_acc = evaluate_model(lr_model, X, y)
        rf_acc = evaluate_model(rf_model, X, y)
        svm_acc = evaluate_model(svm_model, X, y)
        nn_acc = evaluate_nn(nn_model, X_test=X, y_test=y)


        print(lr_pred)
        print(rf_pred)
        print(svm_pred)
        print(nn_pred)
        print(nn_poss)

        print(lr_acc)
        print(rf_acc)
        print(svm_acc)
        print(nn_acc)

        context = {"lr_pred": lr_pred, "rf_pred": rf_pred, "svm_pred": svm_pred, "nn_poss":nn_poss}

        # Render results in HTML template
        return render(request, "result.html", context)
        

    return render(request, "form.html")
