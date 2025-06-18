import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Global variable to store the fitted preprocessor
_preprocessor = None

def preprocess_data(data):
    global _preprocessor
    # Define features and target
    X = data.drop("diabetes", axis=1) if "diabetes" in data.columns else data
    y = data["diabetes"] if "diabetes" in data.columns else None
    
    # Define categorical and numerical columns
    categorical_cols = ["gender", "smoking_history"]
    numerical_cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
    
    # Create preprocessing pipeline
    _preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    # Split data if target is provided
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, None, None, None
    
    # Fit and transform
    X_train = _preprocessor.fit_transform(X_train)
    if X_test is not None:
        X_test = _preprocessor.transform(X_test)
    
    # Get feature names
    cat_features = _preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_features)
    
    return X_train, X_test, y_train, y_test, feature_names

def get_preprocessor():
    """Return the fitted preprocessor."""
    if _preprocessor is None:
        raise ValueError("Preprocessor has not been fitted yet.")
    return _preprocessor