from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    # Initialize and train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model