import importlib
import subprocess

# Check if the required packages are installed
packages = ['numpy', 'scipy', 'pandas', 'matplotlib']
installed_packages = []

for package in packages:
    try:
        importlib.import_module(package)
        installed_packages.append(package)
    except ImportError:
        pass

# Install missing packages using pip if necessary
missing_packages = set(packages) - set(installed_packages)

if missing_packages:
    print(f"Missing packages detected: {', '.join(missing_packages)}")
    print("Installing required packages...")

    for package in missing_packages:
        subprocess.check_call(['pip', 'install', package])

# Import the required packages
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

#constantes
limiteMediaAceptable = 70
valorP_aceptacion = 0.05

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('Conjunto_datos_proyecto3.csv', delimiter=',')

# Exclude "numero muestra" column from data
data = data.drop(columns='Numero muestra')

# Extract the column data
inicial = data['Inicial']
primer_cambio = data['Primer_cambio']
segundo_cambio = data['Segundo_cambio']

def calculate_statistics(data):
    statistics = {}

    for column_name, column_data in data.items():
        column_statistics = {
            'Average': np.mean(column_data),
            'Variance': np.var(column_data),
            'Standard Deviation': np.std(column_data),
            'Median': np.median(column_data),
            '25th Percentile': np.percentile(column_data, 25),
            '50th Percentile (Median)': np.percentile(column_data, 50),
            '75th Percentile': np.percentile(column_data, 75),
            'Mode': stats.mode(column_data)[0][0]
        }
        statistics[column_name] = column_statistics

    return statistics

# Calculate statistics for each column
statistics = calculate_statistics(data)

# Create a DataFrame from the statistics dictionaries
statistics_df = pd.DataFrame(statistics)

# Print the DataFrame
print(statistics_df)

#Pruebas de hipotesis

#Función para comprobar que la muestra analizada tiene un valor de media aceptable para la compañia
# La misma imprime a consola si la muestra es aceptable o no
def perform_statistical_tests(data, dataset_name):

    # Se calculan los valores t y p tomando en cuenta lo siguiente
    # H_{0}: La muestra tiene una media u = 70 (este valor puede cambiarse en la sección de variables)
    # H_{0}: La muestra tiene una media u > 70
    # data se refiere a una de las muestra, puede ser inicial, primer cambio, segundo cambio
    # limiteMediaAceptable hace referencia a la constante que se declara anteriormente
    t_statistic, p_value = stats.ttest_1samp(data, limiteMediaAceptable)

    print("\nPara la muestra", dataset_name + ":")
    if t_statistic > 0:
        print("Dado un valor t =", t_statistic,
              "los datos tienen una media mayor a 70, por lo que se consideran aceptables")
    else:
        print("Dado un valor t =", t_statistic,
              "los datos tienen una media menor a 70, por lo que no se consideran aceptables")

    if p_value < valorP_aceptacion:
        print("Dado un valor p de", p_value, "es menor a", valorP_aceptacion,
              "se puede determinar que los datos son estadísticamente significativos")
    else:
        print("Dado un valor p de", p_value, "es mayor a", valorP_aceptacion,
              "no se puede determinar una diferencia estadísticamente significativa")
        
    return (t_statistic, p_value)

#Comprobando que los datos cumplan con el 70% de media poblacional de rendimiento
print("\nCOMPROBANDO SI TODAS LOS DATOS CUMPLEN CON UN 70 DE MEDIA POBLACIÓN...\n")

perform_statistical_tests(inicial.values, "inicial")
perform_statistical_tests(primer_cambio.values, "primer cambio")
perform_statistical_tests(segundo_cambio.values, "segundo cambio")

#Funcion que compara estadisticamente las medias de las muestras primer cambio y segundo cambio
# e imprime los resultados para cada muestra en consola
def perform_statistical_tests_ind(data1, data2, dataset1_name, dataset2_name):

    #Prueba t que pretende comparar estadisticamente si existe una diferencia significativa
    #en los valores de media entre la muestra inicial, y cualquiera ya sea la primera confi-
    #guración o la segunda
    #data1 hace referencia a la muestra que se va a probar
    #data2 hace referencia a la muestra contra la que se va a contrastar, en este caso siempre es iniciales
    t_statistic, p_value = stats.ttest_ind(data1, data2)

    print("\nCOMPROBANDO DIFERENCIAS ENTRE ", dataset1_name.upper(), " CON INICIAL...")

    if t_statistic > 0:
        print("\nDado un valor t de", t_statistic,
              "se puede determinar que los datos de", dataset1_name, "presentan",
              "un valor de media poblacional mayor al de los datos de", dataset2_name)
    else:
        print("\nDado un valor t de", t_statistic,
              "se puede determinar que los datos de", dataset1_name, "presentan",
              "un valor de media poblacional menor al de los datos de", dataset2_name)

    if p_value < valorP_aceptacion:
        print("\nDado un valor p de", p_value,
              "es menor a", valorP_aceptacion,
              "se puede determinar que hay una diferencia significativa",
              "entre las medias de", dataset1_name, "y", dataset2_name)
    else:
        print("\nDado un valor p de", p_value,
              "es mayor a", valorP_aceptacion,
              "se puede determinar que no hay una diferencia significativa",
              "entre las medias de", dataset1_name, "y", dataset2_name)
        
    return (t_statistic, p_value)
        
print("\nCOMPROBANDO DIFERENCIAS ENTRE PRIMER Y SEGUNDO CAMBIO, CON INICIAL...")

perform_statistical_tests_ind(primer_cambio.values, inicial.values, "primer cambio", "inicial")
perform_statistical_tests_ind(segundo_cambio.values, inicial.values, "segundo cambio", "inicial")

from scipy.stats import norm

optimalSigmas = []

#Haciendo uso de MLE, calcula el parámetro sigma ideal para la distribución normal correcta para aproximar
#las muestras del estudio. Se asume que u (media poblacional) es la misma que la media muestral en cada caso
def calculate_mle_params(data, dataSetDic):

    #Usando una derivada previamente calculada, computa el valor de sigma necesario para maximizar la función
    def getOptimalSigma(data, average):

        sum = 0

        for value in data:
            sum += (value - average)**2
        
        return np.sqrt(sum/len(data))

    result = getOptimalSigma(data, dataSetDic['Average'])

    optimalSigmas.insert(0, result)

    mle_std = result
    estimate_variance = mle_std ** 2

    print("\nPara la muestra:")
    print("MLE Standard Deviation: ", mle_std, " vs ", "Sample Standard Deviation: ", dataSetDic['Standard Deviation'])
    print("MLE Variance: ", estimate_variance, " vs ", "Sample Variance: ", dataSetDic['Variance'])
    print("Error percentage for standard deviation is = ", 100 * (dataSetDic['Standard Deviation'] - mle_std) / dataSetDic['Standard Deviation'])
    print("Error percentage for variance is = ", 100 * (dataSetDic['Variance'] - estimate_variance) / dataSetDic['Variance'])

calculate_mle_params(inicial.values, statistics['Inicial'])
calculate_mle_params(primer_cambio.values, statistics['Primer_cambio'])
calculate_mle_params(segundo_cambio.values, statistics['Segundo_cambio'])

# Display histograms and box plots
for i, (column_name, column_data) in enumerate(data.items()):

    def distribucionNormalEstimada(x_estimatedNormal, sigma, average):

        return 1/(np.sqrt(2*np.pi*(sigma)**2))*np.e**(-(x_estimatedNormal - average)**2/(2*sigma**2))

    x_estimatedNormal = np.linspace(min(column_data), max(column_data))

    #Create a new window for each probability function
    fig_pf = plt.figure(figsize=(8, 6))
    plt.plot(x_estimatedNormal, distribucionNormalEstimada(x_estimatedNormal,
                                                            optimalSigmas.pop(0),
                                                            statistics[str(column_name)]['Average']),
                                                            linewidth=2, label='Normal')
    plt.hist(column_data, bins='auto', density=True)
    plt.title(f'{column_name} Función de probabilidad')
    plt.xlabel('Value')
    plt.ylabel('Probabilidad')
    plt.xlim(45, 100)
    plt.legend()

    # Create a new window for each histogram
    fig_hist = plt.figure(figsize=(8, 6))
    plt.hist(column_data, bins='auto')
    plt.title(f'{column_name} Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(45,100)

    # Create a new window for each box plot
    fig_box = plt.figure(figsize=(8, 6))
    plt.boxplot(column_data, vert=False)
    plt.title(f'{column_name} Box Plot')
    plt.ylabel('Value')
    plt.xlim(45,100)

plt.tight_layout()
plt.show(block=False)

# Wait for user input before finishing execution
input("Press Enter to exit...")