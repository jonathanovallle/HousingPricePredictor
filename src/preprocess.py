import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess(path_to_csv: str):
    df = pd.read_csv(path_to_csv)
    df = df[df['GrLivArea'] < 4500]

    X = df.drop(['SalePrice', 'Id'], axis=1)
    y = df['SalePrice']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    return train_test_split(X_processed, y, test_size=0.2, random_state=42)
