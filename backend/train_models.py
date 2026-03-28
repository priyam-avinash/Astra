import os
import joblib
import logging
from app.services.ai_predictor import ai_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "app", "models", "saved_models")

def train_and_save_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    logger.info("Fetching robust 2-year daily training data (NIFTY 50)...")
    # Using NIFTY as a proxy for broad market dynamics
    df_daily = ai_engine._fetch_data("^NSEI", period="2y", interval="1d")
    
    if df_daily is not None and not df_daily.empty:
        logger.info("Training Astra.ai 1.0 (RandomForestRegressor)...")
        rf_model = ai_engine.train_model_rf(df_daily)
        if rf_model:
            rf_path = os.path.join(MODELS_DIR, "astra_rf.joblib")
            joblib.dump(rf_model, rf_path)
            logger.info(f"Saved RandomForest model to {rf_path}")
        else:
            logger.error("Failed to train RF model.")
    else:
        logger.error("Failed to fetch daily training data.")

    logger.info("Fetching robust 1-year 1-hour training data (NIFTY 50)...")
    # Using 1y for 1h to avoid yfinance limits
    df_hourly = ai_engine._fetch_data("^NSEI", period="730d", interval="1h")
    
    if df_hourly is not None and not df_hourly.empty:
        logger.info("Training Astra.ml (MLPRegressor)...")
        ann_model = ai_engine.train_model_ann(df_hourly)
        if ann_model:
            ann_path = os.path.join(MODELS_DIR, "astra_ann.joblib")
            joblib.dump(ann_model, ann_path)
            logger.info(f"Saved ANN model to {ann_path}")
            
            # Since StandardScaler is attached to model, we save the whole object
        else:
            logger.error("Failed to train ANN model.")
    else:
        logger.error("Failed to fetch hourly training data.")
        
if __name__ == "__main__":
    train_and_save_models()
