import pandas as pd

data_path = "data/processed/features.csv"

def load_data(path=data_path):
    df = pd.read_csv(path)
    return df

def main():
    print("Loading data...")
    
    df = load_data()
    
    print("Data shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()