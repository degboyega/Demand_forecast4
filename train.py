import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os 

def train_model(input_path, model_output_path):
    """
    Trains a Random Forest model to predict gasoline demand and saves the trained model.
    
    Parameters:
    - input_path: Path to the preprocessed dataset (CSV file).
    - model_output_path: Path where the trained model will be saved.
    
    Returns:
    - None
    """

    # Load the preprocessed dataset
    df = pd.read_csv(input_path)

    # Define features (X) and target variable (y)
    X = df.drop(columns=['Gasoline_Demand'])  # Features
    y = df['Gasoline_Demand']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model training complete. Model saved to {model_output_path}")

# Run model training
if __name__ == "__main__":
    # Path to the preprocessed data and output model file
    train_model("processed_data/gasoline_demand_cleaned.csv", "models/gasoline_demand_rf_model.pkl")
