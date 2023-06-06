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

# Extract the column data
inicial = data['Inicial']
primer_cambio = data['Primer_cambio']
segundo_cambio = data['Segundo_cambio']

# Calculate statistics for each column
columns = [inicial, primer_cambio, segundo_cambio]
column_names = ['Inicial', 'Primer_cambio', 'Segundo_cambio']

# Create a dictionary to store the statistics
statistics = {
    'Average': [],
    'Variance': [],
    'Standard Deviation': [],
    'Median': [],
    '25th Percentile': [],
    '50th Percentile (Median)': [],
    '75th Percentile': [],
    'Mode': []
}

info = zip(columns, column_names)

# Calculate statistics for each column
for column, name in info:
    # Convert the column to a NumPy array
    column_data = column.values

    # Calculate statistics using NumPy and SciPy
    average = np.mean(column_data)
    variance = np.var(column_data)
    std_deviation = np.std(column_data)
    median = np.median(column_data)
    quantiles = np.percentile(column_data, [25, 50, 75])
    mode = stats.mode(column_data)[0][0]

    # Add the statistics to the dictionary
    statistics['Average'].append(average)
    statistics['Variance'].append(variance)
    statistics['Standard Deviation'].append(std_deviation)
    statistics['Median'].append(median)
    statistics['25th Percentile'].append(quantiles[0])
    statistics['50th Percentile (Median)'].append(quantiles[1])
    statistics['75th Percentile'].append(quantiles[2])
    statistics['Mode'].append(mode)

# Create a DataFrame from the statistics dictionary
statistics_df = pd.DataFrame(statistics, index=column_names)

# Print the DataFrame
print(statistics_df)

# Create a figure for histograms
fig_hist = plt.figure(figsize=(12, 8))
fig_hist.suptitle('Histograms')

# Create a figure for box plots
fig_box = plt.figure(figsize=(12, 8))
fig_box.suptitle('Box Plots')

# Display histograms and box plots
for i, (column, name) in enumerate(zip(columns, column_names)):
    # Convert the column to a NumPy array
    column_data = column.values

    # Plot histogram
    ax_hist = fig_hist.add_subplot(1, len(columns), i + 1)
    ax_hist.hist(column_data, bins='auto')
    ax_hist.set_title(f'{name} Histogram')
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')

    # Plot box plot
    ax_box = fig_box.add_subplot(1, len(columns), i + 1)
    ax_box.boxplot(column_data)
    ax_box.set_title(f'{name} Box Plot')
    ax_box.set_ylabel('Value')

    # Calculate and add labels to box plot
    box_vals = np.percentile(column_data, [0, 25, 50, 75, 100])
    whiskers = ax_box.get_lines()[:2]
    caps = ax_box.get_lines()[2:4]
    fliers = ax_box.get_lines()[4]

    # Add labels to quantiles
    for j, val in enumerate(box_vals):
        # Adjust x-position offset for quantiles
        x_offset = 0.06 if j == 3 else 0.12
        y_offset = 0.1
        if j != 2:
            ax_box.annotate(f'{val:.2f}', (1 + x_offset, val), xytext=(0, y_offset), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
        else:
            ax_box.annotate(f'{val:.2f}', (1 + x_offset, val), xytext=(0, -y_offset), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))

    # Add labels to whiskers
    whisker_y1 = whiskers[0].get_ydata()[1]
    whisker_y2 = whiskers[1].get_ydata()[1]
    ax_box.annotate(f'{whisker_y1:.2f}', (1.05, whisker_y1), xytext=(0, -10), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
    ax_box.annotate(f'{whisker_y2:.2f}', (1.05, whisker_y2), xytext=(0, 10), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))

    # Add labels to caps
    cap_y1 = caps[0].get_ydata()[1]
    cap_y2 = caps[1].get_ydata()[1]
    ax_box.annotate(f'{cap_y1:.2f}', (1.05, cap_y1), xytext=(0, -10), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
    ax_box.annotate(f'{cap_y2:.2f}', (1.05, cap_y2), xytext=(0, 10), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))

    # Add labels to outliers if available
    if len(fliers.get_ydata()) > 0:
        flier_data = fliers.get_ydata()
        for flier_y in flier_data:
            # Adjust x-position offset for outliers
            x_offset = 0.12
            ax_box.annotate(f'{flier_y:.2f}', (1 + x_offset, flier_y), xytext=(0, -10), textcoords='offset points', ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))

# Show the figures
plt.tight_layout()
plt.show(block=False)

#Pruebas de hipotesis

def perform_statistical_tests(data, dataset_name):

    # Perform one-sample t-test
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

def perform_statistical_tests_ind(data1, data2, dataset1_name, dataset2_name):

    # Perform independent t-test
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

#Obtener los parámetros de ajuste
def calculate_mle_params(data):
    def neg_log_likelihood(params, x):
        sigma, mu = params
        return np.sum(np.log(sigma ** 2) + ((x - mu) ** 2) / sigma ** 2)

    result = opt.minimize(neg_log_likelihood, x0=[1, data.mean()], args=(data,), method='Nelder-Mead')

    mle_std = result.x[0]
    estimate_variance = mle_std ** 2

    print("\nPara la muestra:")
    print("MLE Standard Deviation: ", mle_std, " vs ", "Sample Standard Deviation: ", data.std())
    print("MLE Variance: ", estimate_variance, " vs ", "Sample Variance: ", data.var())
    print("Error percentage for standard deviation is = ", 100 * (data.std() - mle_std) / data.std())
    print("Error percentage for variance is = ", 100 * (data.var() - estimate_variance) / data.var())

calculate_mle_params(inicial.values)
calculate_mle_params(primer_cambio.values)
calculate_mle_params(segundo_cambio.values)

# Wait for user input before finishing execution
input("Press Enter to exit...")
