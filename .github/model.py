import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def train_xgb_model(df):
    feature_cols = [col for col in df.columns if col not in ['target']]
    X = df[feature_cols]
    y = df['target']
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, pred))
    
    print(f"CV Accuracy: {sum(scores)/len(scores):.2%}")
    
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    final_model.fit(X, y)
    return final_model, feature_cols