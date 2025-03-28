import os
import re
import pandas as pd

def parse_metrics(file_path):
    """
    Parse the metrics file and return a dictionary of metric values.
    Extra text (like 'seconds' or '%') is removed.
    """
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split only on the first colon in case values contain colons
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "Training Time":
                    value = value.replace("seconds", "").strip()
                if key == "Mean Absolute Percentage Error (MAPE)":
                    value = value.replace("%", "").strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics

data = []

base_dir = os.getcwd()

pattern = re.compile(r'^(BIDIRECTIONAL|GRU|LSTM)_\d+_\d+$', re.IGNORECASE)

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and pattern.match(folder):
        metrics_file = os.path.join(folder_path, "metrics.txt")
        if os.path.exists(metrics_file):
            metrics = parse_metrics(metrics_file)
            # Expected format: MODEL_EPOCHS_BATCHSIZE
            parts = folder.split("_")
            model = parts[0]
            epochs = int(parts[1])
            batch_size = int(parts[2])

            metrics['Folder'] = folder
            metrics['Model'] = model.upper()
            metrics['Epochs'] = epochs
            metrics['BatchSize'] = batch_size
            data.append(metrics)

df = pd.DataFrame(data)

print("Collected Metrics:")
print(df)


numeric_metrics = ["Training Time", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)",
                   "Mean Absolute Error (MAE)", "Mean Absolute Percentage Error (MAPE)"]
overall_averages = df[numeric_metrics].mean()
print("\nOverall Averages:")
print(overall_averages)

grouped_by_model = df.groupby("Model")[numeric_metrics].mean()
print("\nAverages by Model Type:")
print(grouped_by_model)

sorted_by_mse = df.sort_values("Mean Squared Error (MSE)")
print("\nConfigurations Sorted by MSE (lowest to highest):")
print(sorted_by_mse[['Folder', 'Mean Squared Error (MSE)', 'Training Time']])
