import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Load test data
    digits = load_digits()
    X, y = digits.data, digits.target

    # Load model
    model = joblib.load("model_train.pkl")

    # Predict
    y_pred = model.predict(X)

    # Report
    print("Classification Report:\n")
    print(classification_report(y, y_pred))
