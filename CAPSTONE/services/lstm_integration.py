# services/lstm_integration.py

import os
import pickle
import glob
import shutil
import time
import threading
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import schedule
from flask import current_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
#  LSTM Predictor Class – loads models from a given folder
# ================================================================

class MRT3LSTMPredictor:
    def __init__(self, model_path_pattern='./models_2022-2024_v10_plus_*/'):
        self.model_path_pattern = model_path_pattern
        self.model_path = None
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self.feature_cols = None
        self.station_directions = []

        self.stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
                         "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
                         "Ayala Ave", "Magallanes", "Taft"]
        self.directions = ['Northbound', 'Southbound']

        self._find_latest_model_path()
        if self.model_path:
            self.load_models()
        else:
            logger.warning("No model folder found – models will not be loaded.")

    def _find_latest_model_path(self):
        matching = glob.glob(self.model_path_pattern)
        if matching:
            self.model_path = sorted(matching)[-1]
            logger.info(f"📁 Using latest model folder: {self.model_path}")
        else:
            default = './models_2022-2024_v10_plus_latest/'
            if os.path.exists(default):
                self.model_path = default
                logger.info(f"📁 Using default folder: {default}")
            else:
                self.model_path = None
                logger.error("❌ No model folders found.")

    def load_models(self):
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"❌ Model path not found: {self.model_path}")
            return False

        try:
            feature_cols_path = os.path.join(self.model_path, 'feature_cols.pkl')
            if not os.path.exists(feature_cols_path):
                logger.error(f"❌ feature_cols.pkl missing in {self.model_path}")
                return False

            with open(feature_cols_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            logger.info(f"📋 Loaded {len(self.feature_cols)} feature columns")

            self.models.clear()
            self.feature_scalers.clear()
            self.target_scalers.clear()
            self.station_directions.clear()

            loaded = 0
            for station in self.stations:
                for direction in self.directions:
                    key = f"{station}_{direction}"
                    try:
                        model_file = os.path.join(self.model_path, f'{key}_lstm_v10_plus.keras')
                        fs_file = os.path.join(self.model_path, f'{key}_feature_scaler.pkl')
                        ts_file = os.path.join(self.model_path, f'{key}_target_scaler.pkl')

                        if all(os.path.exists(f) for f in (model_file, fs_file, ts_file)):
                            self.models[key] = load_model(model_file)
                            with open(fs_file, 'rb') as f:
                                self.feature_scalers[key] = pickle.load(f)
                            with open(ts_file, 'rb') as f:
                                self.target_scalers[key] = pickle.load(f)
                            self.station_directions.append(key)
                            loaded += 1
                    except Exception as e:
                        logger.debug(f"⚠️ Could not load {key}: {e}")

            logger.info(f"✅ Loaded {loaded} station-direction models from {self.model_path}")
            return loaded > 0

        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_feature_sequence(self, station, direction, target_datetime, seq_length=24):
        try:
            from services.feature_engineering import get_feature_sequence_for_station
            return get_feature_sequence_for_station(station, direction, target_datetime, seq_length)
        except Exception as e:
            logger.error(f"❌ Error getting feature sequence: {e}")
            return None

    def predict_congestion(self, station, direction, target_datetime=None):
        if target_datetime is None:
            target_datetime = datetime.now()

        key = f"{station}_{direction}"
        if key not in self.models:
            logger.warning(f"⚠️ No model for {key}")
            return None

        try:
            features = self.get_feature_sequence(station, direction, target_datetime)
            if features is None:
                return None

            input_seq = features.reshape(1, 24, -1)
            pred_scaled = self.models[key].predict(input_seq, verbose=0)
            raw = float(pred_scaled[0][0])

            target_scaler = self.target_scalers[key]
            passengers = float(target_scaler.inverse_transform([[raw]])[0][0])

            # Use P95 from the main API (or fallback)
            try:
                from routes.api_predict import get_p95_percentile
                p95 = get_p95_percentile(station, direction)
            except:
                from constants import MRT3_PLATFORM_CAPACITY
                p95 = MRT3_PLATFORM_CAPACITY.get(station, 1000) * 0.8

            congestion = (passengers / p95) * 100
            return max(0, min(100, congestion))

        except Exception as e:
            logger.error(f"❌ Prediction error for {key}: {e}")
            return None

    def get_model_info(self):
        return {
            'total_models': len(self.models),
            'station_directions': self.station_directions,
            'model_path': self.model_path
        }


# ================================================================
#  Global Model Update Helper – replaces the in‑memory models
# ================================================================

def update_global_models(predictor):
    """
    Replace the global directional_models and directional_scalers
    with the models loaded in the predictor.
    """
    try:
        import sys
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, 'directional_models_cached'):
            main_module.directional_models_cached = predictor.models
            main_module.directional_scalers_cached = {
                **predictor.feature_scalers,
                **predictor.target_scalers
            }
            logger.info("✅ Global models updated (in-memory).")
        else:
            import services
            services.directional_models = predictor.models
            services.directional_scalers = {
                **predictor.feature_scalers,
                **predictor.target_scalers
            }
            logger.info("✅ Global models updated (services module).")

        try:
            from flask import current_app
            current_app.config['DIRECTIONAL_MODELS'] = predictor.models
            current_app.config['DIRECTIONAL_SCALERS'] = {
                **predictor.feature_scalers,
                **predictor.target_scalers
            }
        except RuntimeError:
            pass

        return True
    except Exception as e:
        logger.error(f"❌ Failed to update global models: {e}")
        return False


# ================================================================
#  Report‑Based LSTM Training (Local)
# ================================================================

# Feature columns (must match training)
FEATURE_COLS = [
    'TotalPassenger',
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush',
    'is_holiday', 'is_christmas_season', 'is_payday'
]

SEQ_LENGTH = 24
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE_EARLY = 10
PATIENCE_LR = 8

def add_cyclical_time_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df

def add_smart_operating_flags(df):
    time_decimal = df['hour'] + df['minute'] / 60
    df['is_operating_hour'] = ((time_decimal >= 4.5) & (time_decimal < 23.0)).astype(np.int8)
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    return df

def is_holiday(date):
    # Use a simple list of Philippine holidays – you can expand
    holidays = [
        '2022-01-01', '2022-04-09', '2022-04-14', '2022-04-15', '2022-05-01',
        '2022-06-12', '2022-08-21', '2022-08-29', '2022-11-30', '2022-12-08',
        '2022-12-25', '2022-12-30', '2022-12-31'
    ]
    return date.strftime('%Y-%m-%d') in holidays

def is_christmas_season(date):
    mmdd = date.strftime('%m-%d')
    return (mmdd >= '12-15') or (mmdd <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def create_sequences(features, target, seq_length=SEQ_LENGTH):
    n = len(features) - seq_length
    if n <= 0:
        return np.array([]), np.array([])
    X = np.zeros((n, seq_length, features.shape[1]), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        X[i] = features[i:i+seq_length]
        y[i] = target[i+seq_length]
    return X, y

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model

def prepare_report_data(days_back=30, min_reports=50):
    """
    Fetch reports from the database, aggregate by station-direction-hour,
    and return a DataFrame with columns: station, direction, hour_timestamp,
    TotalPassenger (aggregated congestion), and all time features.
    """
    from models import Report, db
    cutoff = datetime.now() - timedelta(days=days_back)

    # Query reports that are not flagged/archived and have valid congestion
    reports = Report.query.filter(
        Report.timestamp >= cutoff,
        Report.reported_congestion.isnot(None),
        Report.flagged == False,
        Report.archived == False
    ).all()

    if len(reports) < min_reports:
        logger.warning(f"⚠️ Only {len(reports)} reports found – need at least {min_reports}.")
        return None

    # Convert to DataFrame
    data = []
    for r in reports:
        data.append({
            'station': r.station,
            'direction': r.direction or 'Northbound',
            'timestamp': r.timestamp,
            'congestion': r.reported_congestion
        })
    df = pd.DataFrame(data)

    # Filter out obvious outliers (e.g., congestion < 0 or > 100)
    df = df[(df['congestion'] >= 0) & (df['congestion'] <= 100)]

    if len(df) < min_reports:
        logger.warning(f"⚠️ After filtering, only {len(df)} reports remain – not enough.")
        return None

    # Group by station, direction, hour
    df['hour_timestamp'] = df['timestamp'].dt.floor('h')
    df['hour'] = df['hour_timestamp'].dt.hour
    df['weekday'] = df['hour_timestamp'].dt.weekday
    df['month'] = df['hour_timestamp'].dt.month
    df['minute'] = df['timestamp'].dt.minute  # not used for grouping but for features

    # Aggregate: we use the average congestion per hour as the "TotalPassenger"
    agg_df = df.groupby(['station', 'direction', 'hour_timestamp']).agg({
        'congestion': 'mean'
    }).rename(columns={'congestion': 'TotalPassenger'}).reset_index()

    # Add time features
    agg_df['hour'] = agg_df['hour_timestamp'].dt.hour
    agg_df['weekday'] = agg_df['hour_timestamp'].dt.weekday
    agg_df['month'] = agg_df['hour_timestamp'].dt.month
    agg_df['minute'] = 0  # all hours start at minute 0

    agg_df = add_cyclical_time_features(agg_df)
    agg_df = add_smart_operating_flags(agg_df)

    # Holiday flags
    agg_df['is_holiday'] = agg_df['hour_timestamp'].apply(is_holiday).astype(np.int8)
    agg_df['is_christmas_season'] = agg_df['hour_timestamp'].apply(is_christmas_season).astype(np.int8)
    agg_df['is_payday'] = agg_df['hour_timestamp'].apply(is_payday).astype(np.int8)

    # Fill missing features if any (should not happen)
    agg_df = agg_df.fillna(0)

    logger.info(f"✅ Prepared {len(agg_df)} hourly aggregated records from {len(reports)} reports.")
    return agg_df

def train_lstm_on_reports():
    """
    Train LSTM models using the aggregated report data.
    Returns the path to the folder where models are saved, or None on failure.
    """
    # 1. Prepare data
    agg_df = prepare_report_data(days_back=90, min_reports=100)  # Use last 90 days, at least 100 reports
    if agg_df is None or len(agg_df) < 100:
        logger.error("❌ Not enough report data to train.")
        return None

    # 2. Loop over each station-direction pair
    model_path = f"./models_2022-2024_v10_plus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_path, exist_ok=True)

    stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
                "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
                "Ayala Ave", "Magallanes", "Taft"]
    directions = ['Northbound', 'Southbound']

    feature_cols = FEATURE_COLS
    with open(os.path.join(model_path, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)

    trained_count = 0

    for station in stations:
        for direction in directions:
            key = f"{station}_{direction}"
            station_df = agg_df[(agg_df['station'] == station) & (agg_df['direction'] == direction)]

            if len(station_df) < SEQ_LENGTH + 20:
                logger.info(f"⏩ Skipping {key} – only {len(station_df)} records.")
                continue

            # Sort by time
            station_df = station_df.sort_values('hour_timestamp')

            # Split chronologically (70% train, 15% val, 15% test)
            n = len(station_df)
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)

            train_df = station_df.iloc[:train_end].copy()
            val_df = station_df.iloc[train_end:val_end].copy()
            test_df = station_df.iloc[val_end:].copy()

            # Scale features
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(train_df[feature_cols])

            # Scale target (congestion values are already 0-100, but we still scale)
            target_scaler = StandardScaler()
            target_scaler.fit(train_df[['TotalPassenger']])

            # Create sequences
            train_features = feature_scaler.transform(train_df[feature_cols])
            train_targets = target_scaler.transform(train_df[['TotalPassenger']]).flatten()
            X_train, y_train = create_sequences(train_features, train_targets)

            # For validation, we combine train+val to get proper sequences
            combined_df = pd.concat([train_df, val_df], axis=0)
            combined_features = feature_scaler.transform(combined_df[feature_cols])
            combined_targets = target_scaler.transform(combined_df[['TotalPassenger']]).flatten()
            X_combined, y_combined = create_sequences(combined_features, combined_targets)

            start_idx = len(train_df) - SEQ_LENGTH
            X_val = X_combined[start_idx:]
            y_val = y_combined[start_idx:]

            if len(X_train) == 0 or len(X_val) == 0:
                logger.info(f"⏩ Skipping {key} – not enough sequences.")
                continue

            # Build and train model
            input_shape = (SEQ_LENGTH, len(feature_cols))
            model = build_lstm_model(input_shape)

            early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY,
                                       restore_best_weights=True, verbose=0)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=PATIENCE_LR, min_lr=0.00001, verbose=0)

            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            # Evaluate on test set
            test_features = feature_scaler.transform(test_df[feature_cols])
            test_targets = target_scaler.transform(test_df[['TotalPassenger']]).flatten()
            X_test, y_test = create_sequences(test_features, test_targets)

            if len(X_test) > 0:
                pred_scaled = model.predict(X_test, verbose=0)
                pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                mae = np.mean(np.abs(pred - actual))
                logger.info(f"✅ {key} – Test MAE: {mae:.2f} (from {len(X_test)} samples)")
            else:
                logger.info(f"✅ {key} – No test data, skipping evaluation.")

            # Save model and scalers
            model.save(os.path.join(model_path, f'{key}_lstm_v10_plus.keras'))
            with open(os.path.join(model_path, f'{key}_feature_scaler.pkl'), 'wb') as f:
                pickle.dump(feature_scaler, f)
            with open(os.path.join(model_path, f'{key}_target_scaler.pkl'), 'wb') as f:
                pickle.dump(target_scaler, f)

            trained_count += 1
            # Clear memory
            del model, history
            tf.keras.backend.clear_session()

    if trained_count == 0:
        logger.error("❌ No models could be trained – not enough data.")
        return None

    logger.info(f"✅ Trained {trained_count} models based on reports. Saved to {model_path}")
    return model_path


# ================================================================
#  Retraining Pipeline (Report‑Based)
# ================================================================

def retrain_and_reload():
    """
    Full retraining pipeline using user reports:
      1. Train models on report data.
      2. Load the new models.
      3. Update global model dictionaries.
    """
    logger.info("🔄 Starting report‑based retraining pipeline...")

    # Step 1: Train on reports
    new_model_folder = train_lstm_on_reports()
    if not new_model_folder:
        logger.error("❌ Retraining failed – no models generated.")
        return False

    # Step 2: Load the new models into a predictor
    predictor = MRT3LSTMPredictor(model_path_pattern=f"{new_model_folder}/")
    predictor.model_path = new_model_folder
    if not predictor.load_models():
        logger.error("❌ Failed to load the new models.")
        return False

    # Step 3: Update global models
    if update_global_models(predictor):
        logger.info(f"✅ Models reloaded from {new_model_folder} and global dictionaries updated.")
        return True
    else:
        logger.error("❌ Failed to update global models.")
        return False


# ================================================================
#  (Optional) Keep Kaggle pipeline as fallback – disabled by default
# ================================================================
#
# KAGGLE_NOTEBOOK = "jupiterd14/mrt-adapt"
# KAGGLE_OUTPUT_DIR = "./kaggle_download"
#
# def trigger_kaggle_retraining():
#     ... # removed for brevity – you can restore if needed.


# ================================================================
#  Scheduler and Flask Integration
# ================================================================

def schedule_weekly_retraining(app):
    """Schedule retraining every Sunday at 3 AM."""
    def weekly_job():
        with app.app_context():
            retrain_and_reload()

    schedule.every().sunday.at("03:00").do(weekly_job)
    logger.info("📅 Weekly retraining scheduled for Sunday 3:00 AM")

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)

    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    return thread


def register_admin_retrain(app):
    """Adds an endpoint to manually trigger retraining."""
    @app.route('/admin/retrain', methods=['POST'])
    def admin_retrain():
        from flask import jsonify
        try:
            success = retrain_and_reload()
            return jsonify({
                'success': success,
                'message': 'Retraining completed' if success else 'Retraining failed'
            }), 200 if success else 500
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


def init_lstm_predictor(app):
    """Lazy‑load the predictor on startup (no‑op)."""
    logger.info("LSTM predictor will be loaded during retraining.")
    return True