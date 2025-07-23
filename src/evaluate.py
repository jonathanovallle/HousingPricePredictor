import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2

def main():
    preprocessor = joblib.load('models/preprocessor.pkl')
    model = joblib.load('models/xgboost.pkl')
    
    test_df = pd.read_csv('data/train.csv')
    y_test = test_df['SalePrice']
    X_test = preprocessor.transform(test_df.drop(columns=['SalePrice', 'Id']))
    
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    main()
