import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from dotenv import load_dotenv
import logging
import joblib

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    raise Exception("Les clés API Binance ne sont pas définies. Veuillez les ajouter au fichier .env.")

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
primary_timeframe = '1h'
secondary_timeframe = '15m'
limit_primary = 500  
limit_secondary = 220  
amount = 0.001  
rsi_period = 14
overbought = 70
oversold = 30
position = None  
simulation = True  
model_path = 'model.joblib'  
start_time = datetime.now()
end_time = start_time + timedelta(days=3)
initial_balance = 1000  
current_balance = initial_balance
btc_balance = 0.0  

logging.basicConfig(
    level=logging.INFO,  
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger()

def fetch_data(symbol, timeframe, limit):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def add_indicators(df, timeframe='primary'):
    df['MA50'] = df['close'].rolling(window=50, min_periods=50).mean()
    df['MA200'] = df['close'].rolling(window=200, min_periods=200).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = down.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    if timeframe == 'secondary':
        df['BB_middle'] = df['close'].rolling(window=20, min_periods=20).mean()
        df['BB_std'] = df['close'].rolling(window=20, min_periods=20).std()
        df['BB_upper_secondary'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower_secondary'] = df['BB_middle'] - 2 * df['BB_std']
        logger.info("Bollinger Bands ajoutés pour le timeframe secondaire.")
    df.dropna(inplace=True)
    logger.debug(f"Colonnes disponibles pour le timeframe '{timeframe}': {df.columns.tolist()}")
    return df

def create_features_labels(primary_df, secondary_df):
    primary_df = primary_df.reset_index()
    secondary_df = secondary_df.reset_index()
    combined_df = pd.merge_asof(
        primary_df,
        secondary_df,
        on='timestamp',
        direction='backward',
        suffixes=('_primary', '_secondary')
    )
    logger.debug(f"Colonnes disponibles après merge_asof: {combined_df.columns.tolist()}")
    combined_df['Future_Close'] = combined_df['close_primary'].shift(-1)
    combined_df['Target'] = (combined_df['Future_Close'] > combined_df['close_primary']).astype(int)
    combined_df.dropna(inplace=True)
    features = [
        'close_primary', 'volume_primary', 'MA50_primary', 'MA200_primary', 'RSI_primary',
        'close_secondary', 'volume_secondary', 'RSI_secondary', 'BB_upper_secondary', 'BB_lower_secondary'
    ]
    missing_features = [feature for feature in features if feature not in combined_df.columns]
    if missing_features:
        logger.error(f"Features manquantes dans le DataFrame combiné: {missing_features}")
        raise KeyError(f"{missing_features} not in index")
    X = combined_df[features]
    y = combined_df['Target']
    return X, y

def train_model(X, y, model_type='random_forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    else:
        raise ValueError("Type de modèle non supporté.")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return model

def save_model(model, path):
    joblib.dump(model, path)
    logger.info(f"Modèle sauvegardé à {path}")

def load_model(path):
    if os.path.exists(path):
        model = joblib.load(path)
        logger.info(f"Modèle chargé depuis {path}")
        return model
    else:
        logger.info("Aucun modèle sauvegardé trouvé. Entraînement d'un nouveau modèle.")
        return None

def get_latest_data(primary_symbol, primary_timeframe, secondary_symbol, secondary_timeframe):
    primary_df = fetch_data(primary_symbol, primary_timeframe, limit_primary)
    primary_df = add_indicators(primary_df, timeframe='primary')
    secondary_df = fetch_data(secondary_symbol, secondary_timeframe, limit_secondary)
    secondary_df = add_indicators(secondary_df, timeframe='secondary')
    primary_df = primary_df.reset_index()
    secondary_df = secondary_df.reset_index()
    combined_latest = pd.merge_asof(
        primary_df.tail(1),
        secondary_df,
        on='timestamp',
        direction='backward',
        suffixes=('_primary', '_secondary')
    )
    if 'BB_upper_secondary' not in combined_latest.columns or 'BB_lower_secondary' not in combined_latest.columns:
        logger.error("Les Bollinger Bands ne sont pas disponibles dans le timeframe secondaire.")
        raise KeyError("Bollinger Bands manquants dans le timeframe secondaire.")
    features = [
        'close_primary', 'volume_primary', 'MA50_primary', 'MA200_primary', 'RSI_primary',
        'close_secondary', 'volume_secondary', 'RSI_secondary', 'BB_upper_secondary', 'BB_lower_secondary'
    ]
    missing_features = [feature for feature in features if feature not in combined_latest.columns]
    if missing_features:
        logger.error(f"Features manquantes dans le DataFrame combiné: {missing_features}")
        raise KeyError(f"{missing_features} not in index")
    latest_features = combined_latest[features]
    logger.debug(f"Colonnes disponibles pour la prédiction: {combined_latest.columns.tolist()}")
    return latest_features

def place_order(signal, symbol, amount):
    global position, current_balance, btc_balance
    ticker = binance.fetch_ticker(symbol)
    last_price = ticker['last']
    if signal == 1 and position != 'long':
        logger.info("Signal d'achat détecté.")
        if not simulation:
            try:
                order = binance.create_market_buy_order(symbol, amount)
                logger.info(f"Ordre d'achat exécuté : {order}")
                current_balance -= last_price * amount
                btc_balance += amount
            except Exception as e:
                logger.error(f"Erreur lors de l'achat : {e}")
        else:
            logger.info(f"[SIMULATION] Achat de {amount} {symbol} à {last_price} USDT")
            current_balance -= last_price * amount
            btc_balance += amount
        position = 'long'
        logger.info(f"Position mise à jour: {position}")
    elif signal == 0 and position == 'long':
        logger.info("Signal de vente détecté.")
        if not simulation:
            try:
                order = binance.create_market_sell_order(symbol, amount)
                logger.info(f"Ordre de vente exécuté : {order}")
                current_balance += last_price * amount
                btc_balance -= amount
            except Exception as e:
                logger.error(f"Erreur lors de la vente : {e}")
        else:
            logger.info(f"[SIMULATION] Vente de {amount} {symbol} à {last_price} USDT")
            current_balance += last_price * amount
            btc_balance -= amount
        position = None
        logger.info(f"Position mise à jour: {position}")
    else:
        logger.info(f"Aucune action requise. Position actuelle: {position}")

def calculate_profit(start_balance, current_balance, btc_balance, current_price):
    total = current_balance + btc_balance * current_price
    profit = total - start_balance
    return profit, total

def run_bot():
    global position, start_time, end_time, current_balance, btc_balance
    model = load_model(model_path)
    if model is None:
        logger.info("Entraînement initial du modèle...")
        primary_df = fetch_data(symbol, primary_timeframe, limit_primary)
        primary_df = add_indicators(primary_df, timeframe='primary')
        secondary_df = fetch_data(symbol, secondary_timeframe, limit_secondary)
        secondary_df = add_indicators(secondary_df, timeframe='secondary')
        X, y = create_features_labels(primary_df, secondary_df)
        model = train_model(X, y, model_type='random_forest')
        save_model(model, model_path)
        logger.info("Modèle entraîné et sauvegardé avec succès.")
    last_train_time = datetime.now()
    retrain_interval = timedelta(days=7)
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            if current_time - last_train_time > retrain_interval:
                logger.info("Réentraînement du modèle...")
                primary_df = fetch_data(symbol, primary_timeframe, limit_primary)
                primary_df = add_indicators(primary_df, timeframe='primary')
                secondary_df = fetch_data(symbol, secondary_timeframe, limit_secondary)
                secondary_df = add_indicators(secondary_df, timeframe='secondary')
                X, y = create_features_labels(primary_df, secondary_df)
                model = train_model(X, y, model_type='random_forest')
                save_model(model, model_path)
                last_train_time = current_time
                logger.info("Modèle réentraîné et sauvegardé avec succès.")
            latest_data = get_latest_data(symbol, primary_timeframe, symbol, secondary_timeframe)
            prediction = model.predict(latest_data)[0]
            action = 'Achat' if prediction == 1 else 'Vente'
            logger.info(f"Prédiction : {action}")
            place_order(prediction, symbol, amount)
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']
            profit, total = calculate_profit(initial_balance, current_balance, btc_balance, current_price)
            logger.info(f"Rentabilité actuelle: {profit:.2f} USDT | Total: {total:.2f} USDT")
            time.sleep(3600)  
        
        except KeyError as e:
            logger.error(f"Erreur de clé lors de la création des features: {e}")
            logger.info("Vérification des données disponibles...")
            logger.info(f"Colonnes disponibles dans primary_df: {primary_df.columns.tolist()}")
            logger.info(f"Colonnes disponibles dans secondary_df: {secondary_df.columns.tolist()}")
            time.sleep(60)  
        except Exception as e:
            logger.error(f"Erreur : {e}")
            time.sleep(60)  
    try:
        final_ticker = binance.fetch_ticker(symbol)
        final_price = final_ticker['last']
        final_profit, final_total = calculate_profit(initial_balance, current_balance, btc_balance, final_price)
        logger.info("----- Rapport de Rentabilité sur 3 Jours -----")
        logger.info(f"Solde initial: {initial_balance} USDT")
        logger.info(f"Solde final: {final_total:.2f} USDT")
        logger.info(f"Profit / Perte: {final_profit:.2f} USDT")
        logger.info("----- Fin du Rapport -----")
        save_model(model, model_path)
    except Exception as e:
        logger.error(f"Erreur lors du calcul final de la rentabilité : {e}")
    exit()

if __name__ == "__main__":
    run_bot()