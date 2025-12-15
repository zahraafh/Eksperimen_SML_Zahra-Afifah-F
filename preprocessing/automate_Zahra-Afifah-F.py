import pandas as pd

def preprocess_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Mengubah kolom TotalCharges dari kategorikal jadi integer
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

    # Handle missing value
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID
    df.drop(columns=["customerID"], inplace=True)

    # One-hot encoding untuk fitur kategorikal
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Konversi boolean ke integer
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Save data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(
        "telco_customer_churn_raw.csv",
        "preprocessing/telco_customer_churn_preprocessing/telco_churn_processed.csv")

