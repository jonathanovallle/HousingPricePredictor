import pandas as pd
import joblib

def main():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/xgboost.pkl")
    test_df = pd.read_csv("data/train.csv")

    X_test = preprocessor.transform(test_df.drop(columns=["Id", "SalePrice"]))
    y_test = test_df["SalePrice"]

    predictions = model.predict(X_test)

    result_df = pd.DataFrame({
        "SalePrice": y_test,
        "Predicted_Price": predictions
    })
    result_df.to_csv("data/predictions.csv", index=False)
    print("✅ Archivo 'data/predictions.csv' generado correctamente.")

if __name__ == "__main__":
    main()
