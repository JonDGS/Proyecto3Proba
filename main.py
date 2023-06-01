import importlib
import subprocess
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

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
import pandas as pd
import matplotlib.pyplot as plt

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

# Calculate statistics for each column
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

# Show the figures
plt.show()

# Wait for user input before finishing execution
input("Press Enter to exit...")
