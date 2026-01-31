import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


#--------------------------------------------------
# Model Constructors
#--------------------------------------------------

def get_logistic_model(random_state=42):
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver = 'lbfgs',        
        )
    
    return model

def get_random_forest_model(random_state=42):
    
    model = RandomForestClassifier(
        max_depth = None,
        random_state = random_state,
        n_jobs = 1,
        n_estimators = 200        
        )
    
    return model

def train_model(model, X, y):
    
    pass

def evaluate_model(model, X, y):
    
    pass
