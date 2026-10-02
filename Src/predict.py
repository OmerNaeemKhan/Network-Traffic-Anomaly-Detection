import os
import sys
import joblib
import pandas as pd
import numpy as np


# ============================================================
# NETWORK TRAFFIC ANOMALY DETECTION - PREDICTION SCRIPT
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Outputs",
    "normal_traffic_isolation_forest.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "Outputs",
    "feature_scaler.pkl"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "Outputs",
    "prediction_results.csv"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "Raw",
    "UNSW_NB15_testing-set.csv"
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

def load_model_and_scaler():

    print("=" * 60)
    print("LOADING MODEL AND SCALER")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Model loaded successfully!")
    print("Scaler loaded successfully!")

    return model, scaler


# ============================================================
# LOAD DATA
# ============================================================

def load_data(data_path):

    print("\n" + "=" * 60)
    print("LOADING NETWORK TRAFFIC DATA")
    print("=" * 60)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully!")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {df.shape[1]}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)

    data = df.copy()

    # --------------------------------------------------------
    # Remove ONLY target columns.
    #
    # IMPORTANT:
    # DO NOT REMOVE 'id'
    # The trained scaler expects 'id' as a feature.
    # --------------------------------------------------------

    columns_to_remove = [
        "attack_cat",
        "label"
    ]

    for column in columns_to_remove:

        if column in data.columns:
            data = data.drop(columns=[column])

            print(f"Removed target column: {column}")

    # --------------------------------------------------------
    # Convert categorical columns to numeric
    # --------------------------------------------------------

    categorical_columns = data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print(
        f"Categorical columns found: "
        f"{len(categorical_columns)}"
    )

    for column in categorical_columns:

        data[column] = pd.factorize(
            data[column]
        )[0]

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    data = data.fillna(0)

    print(
        f"Features after preprocessing: "
        f"{data.shape[1]}"
    )

    return data


# ============================================================
# ALIGN FEATURES EXACTLY WITH TRAINING FEATURES
# ============================================================

def align_features(features_df, scaler):

    print("\n" + "=" * 60)
    print("ALIGNING FEATURES WITH TRAINED MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Get exact feature names used during training
    # --------------------------------------------------------

    if hasattr(scaler, "feature_names_in_"):

        expected_features = list(
            scaler.feature_names_in_
        )

        print(
            f"Exact training features found: "
            f"{len(expected_features)}"
        )

    else:

        raise ValueError(
            "The saved scaler does not contain the original "
            "feature names. Please retrain the model and scaler "
            "using a DataFrame with column names."
        )

    current_features = list(
        features_df.columns
    )

    print(
        f"Current data features: "
        f"{len(current_features)}"
    )

    # --------------------------------------------------------
    # Find missing features
    # --------------------------------------------------------

    missing_features = [

        feature

        for feature in expected_features

        if feature not in features_df.columns

    ]

    # --------------------------------------------------------
    # Find extra features
    # --------------------------------------------------------

    extra_features = [

        feature

        for feature in features_df.columns

        if feature not in expected_features

    ]

    # --------------------------------------------------------
    # Display missing features
    # --------------------------------------------------------

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print(f" - {feature}")

    else:

        print(
            "\nNo required features are missing."
        )

    # --------------------------------------------------------
    # Display extra features
    # --------------------------------------------------------

    if extra_features:

        print("\nExtra features removed:")

        for feature in extra_features:
            print(f" - {feature}")

    else:

        print(
            "\nNo extra features found."
        )

    # --------------------------------------------------------
    # Add genuinely missing training features
    #
    # This should normally not happen when using the UNSW-NB15
    # dataset, but zero is used as a fallback.
    # --------------------------------------------------------

    for feature in missing_features:

        features_df[feature] = 0

    # --------------------------------------------------------
    # CRITICAL STEP:
    # Reorder columns into EXACTLY the same order used in training
    # --------------------------------------------------------

    features_df = features_df[
        expected_features
    ]

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\nFinal feature count:")
    print(features_df.shape[1])

    print(
        f"Model expects: "
        f"{scaler.n_features_in_} features"
    )

    if features_df.shape[1] != scaler.n_features_in_:

        raise ValueError(
            "FEATURE ALIGNMENT FAILED!"
        )

    print(
        "Feature names and order match training data!"
    )

    return features_df


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(features_df, scaler):

    print("\n" + "=" * 60)
    print("SCALING FEATURES")
    print("=" * 60)

    scaled_features = scaler.transform(
        features_df
    )

    print(
        "Features scaled successfully!"
    )

    return scaled_features


# ============================================================
# RUN ANOMALY DETECTION
# ============================================================

def run_prediction(model, scaled_features):

    print("\n" + "=" * 60)
    print("RUNNING ANOMALY DETECTION")
    print("=" * 60)

    # Isolation Forest:
    #
    #  1  = Normal
    # -1  = Anomaly

    predictions = model.predict(
        scaled_features
    )

    anomaly_scores = model.decision_function(
        scaled_features
    )

    # Convert:
    #
    # 0 = Normal
    # 1 = Anomaly

    predicted_labels = np.where(
        predictions == -1,
        1,
        0
    )

    print(
        "Anomaly detection completed!"
    )

    return predicted_labels, anomaly_scores


# ============================================================
# CREATE RESULTS
# ============================================================

def create_results(
    original_df,
    predicted_labels,
    anomaly_scores
):

    print("\n" + "=" * 60)
    print("CREATING RESULTS")
    print("=" * 60)

    results_df = original_df.copy()

    results_df["Predicted_Label"] = (
        predicted_labels
    )

    results_df["Prediction"] = np.where(
        predicted_labels == 1,
        "Anomaly",
        "Normal"
    )

    results_df["Anomaly_Score"] = (
        anomaly_scores
    )

    print(
        "Results created successfully!"
    )

    return results_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results_df):

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("Results saved successfully!")
    print(f"Location: {RESULTS_PATH}")


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(results_df):

    total_records = len(results_df)

    anomalies = int(
        (
            results_df["Predicted_Label"] == 1
        ).sum()
    )

    normal = int(
        (
            results_df["Predicted_Label"] == 0
        ).sum()
    )

    anomaly_percentage = (
        anomalies / total_records
    ) * 100

    normal_percentage = (
        normal / total_records
    ) * 100

    print("\n")
    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(
        f"Total records analyzed: "
        f"{total_records:,}"
    )

    print(
        f"Normal traffic: "
        f"{normal:,} "
        f"({normal_percentage:.2f}%)"
    )

    print(
        f"Anomalies detected: "
        f"{anomalies:,} "
        f"({anomaly_percentage:.2f}%)"
    )

    print("\nResults saved to:")
    print(RESULTS_PATH)

    print("=" * 60)
    print(
        "ANOMALY DETECTION COMPLETED SUCCESSFULLY!"
    )
    print("=" * 60)


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_network_traffic(data_path):

    print("\n")
    print("=" * 60)
    print("NETWORK TRAFFIC ANOMALY DETECTION")
    print("=" * 60)

    # Step 1: Load model and scaler
    model, scaler = load_model_and_scaler()

    # Step 2: Load data
    original_df = load_data(
        data_path
    )

    # Step 3: Prepare features
    features_df = prepare_features(
        original_df
    )

    # Step 4: Match exact training features
    features_df = align_features(
        features_df,
        scaler
    )

    # Step 5: Scale features
    scaled_features = scale_features(
        features_df,
        scaler
    )

    # Step 6: Predict anomalies
    predicted_labels, anomaly_scores = (
        run_prediction(
            model,
            scaled_features
        )
    )

    # Step 7: Create results
    results_df = create_results(
        original_df,
        predicted_labels,
        anomaly_scores
    )

    # Step 8: Save results
    save_results(
        results_df
    )

    # Step 9: Display summary
    display_summary(
        results_df
    )

    return results_df


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    try:

        results = predict_network_traffic(
            DATA_PATH
        )

    except Exception as error:

        print("\n")
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(error)

        print("=" * 60)

        sys.exit(1)