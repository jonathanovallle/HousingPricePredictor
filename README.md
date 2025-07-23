# 🏠 Housing Price Predictor

Este proyecto utiliza machine learning para predecir el precio de viviendas a partir de sus características. Está basado en el conocido dataset **Ames Housing** y utiliza un modelo de regresión XGBoost junto con un pipeline de preprocesamiento.

## 📁 Estructura del proyecto

```
HousingPricePredictor/
├── data/               # Datos de entrenamiento, prueba y nuevos para predecir
│   ├── train.csv
│   ├── test.csv
│   ├── new_houses.csv
│   └── predictions.csv
├── models/             # Modelos entrenados (.pkl)
│   ├── preprocessor.pkl
│   └── xgboost.pkl
├── src/                # Código fuente
│   ├── preprocess.py   # Carga y preprocesamiento
│   ├── train.py        # Entrenamiento del modelo
│   ├── evaluate.py     # Evaluación del modelo
│   └── visualize.py    # Visualización de resultados
|   |__ predict.py      # Hace la prediccion con los datos de new_houses.csv
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Este archivo
```

## ⚙️ Requisitos

- Python 3.10+
- pip

## 🚀 Instrucciones de uso

1. **Clona el repositorio**:

```bash
git clone https://github.com/jonathanovallle/HousingPricePredictor.git
cd HousingPricePredictor
```

2. **Crea un entorno virtual**:

```bash
python -m venv venv
venv\Scripts\activate  # En Windows
```

3. **Instala las dependencias**:

```bash
pip install -r requirements.txt
```

4. **Ejecuta el flujo completo**:

```bash
# Procesa los datos del csv
python -m src.preprocess

# Entrenamiento del modelo
python -m src.train

# Evaluación del modelo en train.csv
python -m src.evaluate

# Predice el precio del new_houses.csv
python -m src.predict

# Visualización de resultados
python -m src.visualize
```

## 🧠 Modelo

- **Tipo**: Regressor (XGBoost)
- **Entrenamiento**: Pipeline de preprocesamiento + modelo
- **Evaluación**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score

## 📊 Ejemplo de predicciones

```
Predicted_Price
0    201463.36
1    178592.78
2    220655.25
3    143025.63
4    255786.23
```

## 📈 Visualización

Se genera un gráfico de dispersión con los precios reales vs. los predichos en el conjunto de test para facilitar la evaluación visual del modelo.

## 🧪 Dataset

El proyecto utiliza el [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) y ha sido particionado manualmente en:

- `train.csv`
- `test.csv`
- `new_houses.csv` (datos sin precios, para predicción real)

## 🛠️ Herramientas y librerías

- [scikit-learn](https://scikit-learn.org/)
- [xgboost](https://xgboost.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [joblib](https://joblib.readthedocs.io/)

## 👤 Autor

Proyecto desarrollado por [Jonathan Ovalle](https://github.com/jonathanovallle)

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License**.
