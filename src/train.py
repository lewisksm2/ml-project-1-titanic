import pandas as pd

from sklearn.model_selection import train_test_split

from model import (
    get_logistic_model,
    get_random_forest_model,
    train_model,
    evaluate_model
    )

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

def print_results(name, results):
    print(f"\n=== {name} Results ===")
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    if results["roc_auc"] is not None:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

def main():
    print("Loading data...")
    
    df = load_data()
    
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    
    print("Training models...")
    
    log_model = get_logistic_model()
    rf_model = get_random_forest_model()

    log_model = train_model(log_model, X_train, y_train)
    rf_model = train_model(rf_model, X_train, y_train)
    
    print("Evaluating models...")
    
    log_results = evaluate_model(log_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)

    print_results("Logistic Regression", log_results)
    print_results("Random Forest", rf_results)


if __name__ == "__main__":
    main()