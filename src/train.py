import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.preprocessing import load_and_preprocess

def train_model(X_train, y_train, model_type='xgboost'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'lasso':
        model = Lasso()
    elif model_type == 'rf':
        model = RandomForestRegressor()
    else:
        model = XGBRegressor()
    
    model.fit(X_train, y_train)
    return model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess('data/train.csv')
    
    model = train_model(X_train, y_train, model_type='xgboost')
    
    joblib.dump(model, 'models/xgboost.pkl')
    print("Modelo entrenado y guardado en models/xgboost.pkl")

if __name__ == "__main__":
    main()
