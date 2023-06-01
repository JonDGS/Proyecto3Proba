import numpy as np
import scipy.stats as stats
import pandas as pd

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

# Calculate statistics using NumPy and SciPy and store the results in the dictionary
for column, name in zip(columns, column_names):
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
