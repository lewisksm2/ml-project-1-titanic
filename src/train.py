import pandas as pd

data_path = "data/processed/features.csv"
target_col = "Survived"

def load_data(path=data_path):
    df = pd.read_csv(path)
    return df

def split_features_target(df):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def main():
    print("Loading data...")
    
    df = load_data()
    
    X, y = split_features_target(df)
    
    print("X Shape:", X.shape)
    print("y Shape:", y.shape)


if __name__ == "__main__":
    main()