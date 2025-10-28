import os
import time
import ccxt
from dotenv import load_dotenv
from data import fetch_btc_data, create_features
from model import train_xgb_model

load_dotenv()

class BTCBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
        })
        self.symbol = 'BTC/USDT'
        self.model = None
        self.feature_cols = None
        self.last_train = 0

    def ensure_model(self):
        current_hour = int(time.time() // 3600)
        if self.model is None or current_hour - self.last_train > 24 * 7:
            print("🔄 Retraining model...")
            df = fetch_btc_data(limit=2000)
            df_feat = create_features(df)
            self.model, self.feature_cols = train_xgb_model(df_feat)
            self.last_train = current_hour

    def get_signal(self):
        df = fetch_btc_data(limit=200)
        df_feat = create_features(df)
        latest = df_feat[self.feature_cols].iloc[[-1]]
        return self.model.predict(latest)[0]

    def run(self):
        print("🤖 BTC AI Trading Bot (Paper Mode) - Started")
        while True:
            try:
                self.ensure_model()
                signal = self.get_signal()
                price = self.exchange.fetch_ticker(self.symbol)['last']
                
                if signal == 1:
                    balance = self.exchange.fetch_balance()
                    usdt = float(balance['USDT']['free'])
                    if usdt > 10:
                        amount = (usdt * 0.9) / price
                        print(f"🟢 SIGNAL: Mua {amount:.6f} BTC tại {price}")
                        # self.exchange.create_market_buy_order(self.symbol, amount)
                else:
                    print(f"🔴 NO TRADE | BTC: {price}")
                
                time.sleep(3600)
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = BTCBot()
    bot.run()