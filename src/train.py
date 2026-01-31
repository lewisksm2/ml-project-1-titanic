import pandas as pd

from sklearn.model_selection import train_test_split

data_path = "data/processed/features.csv"
target_col = "Survived"
random_state = 42

def load_data(path=data_path):
    df = pd.read_csv(path)
    return df

def split_features_target(df):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def make_train_test_split(X, y, test_size=0.2):
    return train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify=y        
        )

def main():
    print("Loading data...")
    
    df = load_data()
    
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    
    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])


if __name__ == "__main__":
    main()