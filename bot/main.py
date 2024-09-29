import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import os

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# Configuration de l'API Binance
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})
# Paramètres
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 1000  # Nombre de bougies à récupérer
amount = 0.001  # Quantité de BTC à acheter/vendre

def fetch_data(symbol, timeframe, limit):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def add_indicators(df):
    # Moyennes Mobiles
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # Suppression des valeurs manquantes
    df.dropna(inplace=True)
    return df

def create_features_labels(df):
    df['Future_Close'] = df['close'].shift(-1)
    df['Target'] = (df['Future_Close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    
    # Mettre à jour les features pour inclure les nouveaux indicateurs
    features = ['close', 'volume', 'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
    X = df[features]
    y = df['Target']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def get_latest_data(symbol, timeframe):
    df = fetch_data(symbol, timeframe, limit=100)
    df = add_indicators(df)
    X_latest = df[['close', 'volume', 'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']].iloc[-1].values.reshape(1, -1)
    return X_latest

def place_order(signal, symbol, amount):
    if signal == 1:
        print("Signal d'achat détecté")
        # binance.create_market_buy_order(symbol, amount)
    elif signal == 0:
        print("Signal de vente détecté")
        # binance.create_market_sell_order(symbol, amount)

# Pipeline complet avec données en direct via l'API
df = fetch_data(symbol, timeframe, limit)
df = add_indicators(df)
X, y = create_features_labels(df)
model = train_model(X, y)

# Configuration pour l'entraînement régulier
last_train_time = datetime.now()
retrain_interval = timedelta(days=7)  # Réentraîner toutes les 7 jours

# Boucle principale du bot
while True:
    try:
        current_time = datetime.now()
        # Vérifier si c'est le moment de réentraîner le modèle
        if current_time - last_train_time > retrain_interval:
            print("Réentraînement du modèle...")
            df = fetch_data(symbol, timeframe, limit)
            df = add_indicators(df)
            X, y = create_features_labels(df)
            model = train_model(X, y)
            last_train_time = current_time
            print("Modèle réentraîné avec succès.")
        
        # Récupérer les dernières données et faire une prédiction
        X_latest = get_latest_data(symbol, timeframe)
        prediction = model.predict(X_latest)[0]
        place_order(prediction, symbol, amount)
        time.sleep(3600)  # Attendre 1 heure
    
    except Exception as e:
        print(f"Erreur: {e}")
        time.sleep(60)  # Attendre 1 minute avant de réessayer
