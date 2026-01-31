import pandas as pd
import numpy as np

def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Fare"] = np.log1p(df["Fare"])
    
    df = df.dropna(subset = ["Embarked"])
    df = pd.get_dummies(df, columns = ["Embarked"], drop_first=True)

    df = df.drop(columns = ["Cabin", "Ticket", "Name", "PassengerId"])
    
    

    return(df)



clean_data(load_raw_data("data/raw/titanic.csv"))