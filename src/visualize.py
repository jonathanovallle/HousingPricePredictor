import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Cargar el archivo con predicciones y precios reales
    df = pd.read_csv("data/predictions.csv")

    if 'SalePrice' not in df.columns or 'Predicted_Price' not in df.columns:
        raise ValueError("El archivo debe tener las columnas 'SalePrice' y 'Predicted_Price'")

    # Gráfico de dispersión: real vs predicho
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='SalePrice', y='Predicted_Price', data=df)
    plt.plot([df['SalePrice'].min(), df['SalePrice'].max()],
             [df['SalePrice'].min(), df['SalePrice'].max()],
             color='red', linestyle='--', label='Línea ideal')
    plt.title("Precios Reales vs Predichos")
    plt.xlabel("Precio Real")
    plt.ylabel("Precio Predicho")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histograma de errores
    df['Error'] = df['Predicted_Price'] - df['SalePrice']
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Error'], bins=30, kde=True)
    plt.title("Distribución del Error de Predicción")
    plt.xlabel("Error (Predicho - Real)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
