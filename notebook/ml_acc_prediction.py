import os
import joblib
import json
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def kmh_to_ms(speed_kmh):
    return speed_kmh * (1000 / 3600)

def update_dict_from_file(path, key, value):

    # Load existing dictionary or initialize empty one
    if os.path.exists(path):
        with open(path, 'r') as f:
            dictionary = json.load(f)
    else:
        dictionary = {}

    # Update or add entry
    if key in dictionary:
        print(f"Key '{key}' exists. Updating value to '{value}'.")
    else:
        print(f"Key '{key}' not found. Adding key with value '{value}'.")

    dictionary[key] = value

    # Save back to file
    with open(path, 'w') as f:
        json.dump(dictionary, f, indent=4)

    print("Dictionary successfully updated and saved.")


if __name__ == '__main__':
    # Load your dataset
    gap_settings = ['Medium','Long','XLong']

    model_name = "rf_acc"
    for gap_setting in tqdm(gap_settings):
        df = pd.read_csv("../data/combined_data.csv")
        df['Speed Follower'] = kmh_to_ms(df['Speed Follower'])
        df['Speed Leader'] = kmh_to_ms(df['Speed Leader'])
        df = df[df['gap_setting']==gap_setting]

        # Compute relative speed and spacing
        df["delta_v"] = df["Speed Follower"] - df["Speed Leader"]

        # Estimate time step and acceleration
        df["dt"] = df["Time"].diff()
        df["acc_follower"] = df["Speed Follower"].diff() / df["dt"]

        # Estimate time step and acceleration
        df["dt"] = df["Time"].diff()
        df["acc_follower"] = df["Speed Follower"].diff() / df["dt"]

        # Drop the first row (NaN)
        df = df.dropna(subset=["dt", "acc_follower"])

        # Copy the cleaned DataFrame so we can work safely
        df_extended = df.copy()

        # One-hot encode the 'gap_setting' column
        gap_dummies = pd.get_dummies(df_extended['gap_setting'], prefix='gap')

        # Combine all selected features
        features = pd.concat([
            df_extended[["delta_v", "Speed Follower", "speed_fluctuation", "Spacing"]],
            gap_dummies
        ], axis=1)

        # Keep the same target: spacing
        target = df_extended["acc_follower"]



        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf_model.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(rf_model, f"rf_model_acceleration_{gap_setting}.pkl")
        # Predict and evaluate
        y_pred = rf_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)

        model_gap = f"{model_name}_{gap_setting}"
        update_dict_from_file("../REPORTS/rmse.json", model_gap, rmse)

        # r2 = r2_score(y_test, y_pred)

        # rmse, r2

        # import joblib
        #
        # # Load the trained model from the file
        # rf_model_loaded = joblib.load("rf_model_spacing.pkl")
        #
        # prediction = rf_model_loaded.predict(features)





