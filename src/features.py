import pandas as pd
import numpy as np

def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame, output_path = None) -> pd.DataFrame:
    df = df.copy()
    
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Fare"] = np.log1p(df["Fare"])
    
    df = df.dropna(subset = ["Embarked"])
    df = pd.get_dummies(df, columns = ["Embarked"], drop_first=True)

    df = df.drop(columns = ["Cabin", "Ticket", "Name", "PassengerId"])
    
    if output_path:
        df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    df = load_raw_data("data/raw/titanic.csv")
    clean_df = clean_data(df, "data/processed/features.csv")
    print(f"Data successfully cleaned and saved to: data/processed/features.csv")
    




