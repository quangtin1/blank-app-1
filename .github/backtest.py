import vectorbt as vbt
from data import fetch_btc_data, create_features
from model import train_xgb_model

def run_backtest():
    df = fetch_btc_data(limit=1500)
    df_feat = create_features(df)
    model, feature_cols = train_xgb_model(df_feat)
    
    X = df_feat[feature_cols]
    predictions = model.predict(X)
    
    entries = predictions == 1
    exits = ~entries
    
    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        fees=0.001,
        freq='1h'
    )
    
    print(portfolio.stats())
    portfolio.plot().show()
    return portfolio

if __name__ == "__main__":
    run_backtest()