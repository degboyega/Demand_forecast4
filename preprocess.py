import os
import pandas as pd

def preprocess_data(input_path, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(input_path)

    # Ensure that the 'Date' column is of datetime type
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Sort the dataframe by 'Date'
    df.sort_values("Date", inplace=True)

    # Reset index after sorting
    df.reset_index(drop=True, inplace=True)

    # --- Time-based features ---
    df['Day_of_Week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Month'] = df['Date'].dt.month

    # --- Lag features (previous days' demand) ---
    df['Lag_1'] = df['Gasoline_Demand'].shift(1)
    df['Lag_7'] = df['Gasoline_Demand'].shift(7)
    df['Lag_14'] = df['Gasoline_Demand'].shift(14)

    # --- Rolling mean features ---
    df['RollingMean_7'] = df['Gasoline_Demand'].rolling(window=7).mean()
    df['RollingMean_30'] = df['Gasoline_Demand'].rolling(window=30).mean()

    # Drop the 'Date' column as it's no longer needed
    df.drop(columns='Date', inplace=True)

    # Drop rows with missing values (resulting from shifts and rolling windows)
    df.dropna(inplace=True)

    # Save the preprocessed data to the output path
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Data saved to {output_path}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data("data/fuel_demand_data.csv", "processed_data/gasoline_demand_cleaned.csv.csv")
