import pandas as pd
import joblib

def main():
    # Cargar el preprocesador y el modelo entrenado
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/xgboost.pkl")

    # Cargar el conjunto de prueba
    test_df = pd.read_csv("data/train.csv")

    # Extraer variables
    X_test = preprocessor.transform(test_df.drop(columns=["Id", "SalePrice"]))
    y_test = test_df["SalePrice"]

    # Predecir
    predictions = model.predict(X_test)

    # Crear nuevo DataFrame con SalePrice y Predicted_Price
    result_df = pd.DataFrame({
        "SalePrice": y_test,
        "Predicted_Price": predictions
    })

    # Guardar el archivo para visualización
    result_df.to_csv("data/predictions.csv", index=False)
    print("✅ Archivo 'data/predictions.csv' generado correctamente.")

if __name__ == "__main__":
    main()
