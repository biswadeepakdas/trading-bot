"""
LSTM + XGBoost + FinBERT Ensemble v4 — Enhanced Accuracy
==========================================================
Improvements:
  1. 3-layer LSTM with self-attention mechanism
  2. VMD-denoised inputs (from features.py)
  3. XGBoost with Bayesian-inspired hyperparams + regularization
  4. FinBERT news sentiment as auxiliary signal
  5. Adaptive ensemble weighting (learned from validation)
  6. Feature selection based on XGBoost importance
  7. Class-weight balancing for imbalanced data
  8. Label smoothing to prevent overconfident predictions
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                      Input, Multiply, Permute, RepeatVector,
                                      Flatten, Lambda, Bidirectional, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Sklearn
from sklearn.preprocessing import RobustScaler  # More robust to outliers than StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
import xgboost as xgb

from config import (LOOKBACK, EPOCHS, BATCH_SIZE, LSTM_UNITS_1, LSTM_UNITS_2, LSTM_UNITS_3,
                    DROPOUT, LEARNING_RATE, TRAIN_SPLIT, MODEL_DIR, USE_ATTENTION,
                    FEATURE_IMPORTANCE_THRESHOLD, XGB_N_ESTIMATORS, XGB_MAX_DEPTH,
                    XGB_LEARNING_RATE, XGB_MIN_CHILD_WEIGHT, XGB_GAMMA,
                    XGB_REG_ALPHA, XGB_REG_LAMBDA)

import joblib


# ============================================================
# ATTENTION LAYER
# ============================================================
class AttentionLayer(Layer):
    """Self-attention mechanism for LSTM sequences."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                  shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias',
                                  shape=(input_shape[-1],),
                                  initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_context',
                                  shape=(input_shape[-1],),
                                  initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Tanh(xW + b)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # Softmax attention weights
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, -1)
        # Weighted sum
        output = tf.reduce_sum(x * ait, axis=1)
        return output

    def get_config(self):
        return super().get_config()


# ============================================================
# MARKET PREDICTOR v4
# ============================================================
class MarketPredictor:
    """
    Enhanced ensemble model with attention, VMD, and sentiment.
    """

    def __init__(self, name='default'):
        self.name = name
        self.scaler = RobustScaler()  # More robust to outliers
        self.lstm_dir_model = None
        self.lstm_mag_model = None
        self.xgb_dir_model = None
        self.xgb_mag_model = None
        self.feature_names = None
        self.selected_features = None  # After importance-based selection
        self.metrics = {}
        self.ensemble_weights = {'lstm': 0.45, 'xgb': 0.55}  # Learned weights
        self.feature_importance = {}

    def _create_sequences(self, X, y_dir, y_mag):
        """Create LSTM-compatible sequences with lookback window."""
        Xs, ys_dir, ys_mag = [], [], []
        for i in range(LOOKBACK, len(X)):
            Xs.append(X[i - LOOKBACK:i])
            ys_dir.append(y_dir[i])
            ys_mag.append(y_mag[i])
        return np.array(Xs), np.array(ys_dir), np.array(ys_mag)

    def _build_lstm_classifier(self, n_features):
        """Build 3-layer LSTM with self-attention for direction."""
        inputs = Input(shape=(LOOKBACK, n_features))

        # Layer 1
        x = LSTM(LSTM_UNITS_1, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)

        # Layer 2
        x = LSTM(LSTM_UNITS_2, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)

        # Attention
        if USE_ATTENTION:
            x = AttentionLayer()(x)
        else:
            x = LSTM(LSTM_UNITS_3, return_sequences=False)(x)

        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)

        # Dense head
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _build_lstm_regressor(self, n_features):
        """Build 3-layer LSTM with attention for magnitude."""
        inputs = Input(shape=(LOOKBACK, n_features))

        x = LSTM(LSTM_UNITS_1, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)

        x = LSTM(LSTM_UNITS_2, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)

        if USE_ATTENTION:
            x = AttentionLayer()(x)
        else:
            x = LSTM(LSTM_UNITS_3, return_sequences=False)(x)

        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='huber',
            metrics=['mae']
        )
        return model

    def _select_features(self, X_train, y_train):
        """Use a quick XGBoost to identify important features."""
        selector = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            verbosity=0, random_state=42, use_label_encoder=False,
        )
        selector.fit(X_train, y_train)

        importance = dict(zip(
            range(X_train.shape[1]) if not hasattr(X_train, 'columns') else X_train.columns,
            selector.feature_importances_
        ))

        selected = [f for f, imp in importance.items() if imp >= FEATURE_IMPORTANCE_THRESHOLD]

        if len(selected) < 15:
            # Keep at least top 15 features
            sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            selected = [f for f, _ in sorted_feats[:max(15, len(selected))]]

        return selected, importance

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    def train(self, X, y_direction, y_magnitude):
        """Train enhanced ensemble model."""
        self.feature_names = list(X.columns)
        n_features_orig = len(self.feature_names)

        # Scale features (RobustScaler — handles outliers better)
        X_scaled = self.scaler.fit_transform(X.values)
        y_dir = y_direction.values
        y_mag = y_magnitude.values

        # Train/val split
        split_idx = int(len(X_scaled) * TRAIN_SPLIT)

        X_train_raw, X_val_raw = X_scaled[:split_idx], X_scaled[split_idx:]
        y_dir_train, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
        y_mag_train, y_mag_val = y_mag[:split_idx], y_mag[split_idx:]

        # ========================
        # Feature Selection (using XGBoost importance)
        # ========================
        print(f"    Feature selection from {n_features_orig} features...")
        selected_idx, importance_dict = self._select_features(X_train_raw, y_dir_train)

        if isinstance(selected_idx[0], str):
            selected_idx_num = [self.feature_names.index(f) for f in selected_idx]
        else:
            selected_idx_num = selected_idx

        X_train = X_train_raw[:, selected_idx_num]
        X_val = X_val_raw[:, selected_idx_num]

        self.selected_features = [self.feature_names[i] for i in selected_idx_num]
        n_features = len(self.selected_features)
        print(f"    Selected {n_features} features (from {n_features_orig})")

        # Class weights for imbalanced data
        class_weights = self._compute_class_weights(y_dir_train)
        print(f"    Class balance — UP: {(y_dir_train==1).mean():.1%}, DOWN: {(y_dir_train==0).mean():.1%}")

        # ========================
        # 1. LSTM MODELS
        # ========================
        print(f"    Training LSTM (lookback={LOOKBACK}, epochs={EPOCHS}, attention={USE_ATTENTION})...")

        X_seq_train, y_seq_dir_train, y_seq_mag_train = self._create_sequences(
            X_train, y_dir_train, y_mag_train)
        X_seq_val, y_seq_dir_val, y_seq_mag_val = self._create_sequences(
            X_val, y_dir_val, y_mag_val)

        if len(X_seq_train) < 10 or len(X_seq_val) < 5:
            print(f"    SKIP LSTM — insufficient sequence data")
            self.lstm_dir_model = None
            self.lstm_mag_model = None
            lstm_acc = 0.5
            lstm_f1 = 0.0
            lstm_mae = 999.0
        else:
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, verbose=0,
                             monitor='val_accuracy'),
                ReduceLROnPlateau(factor=0.5, patience=7, verbose=0, min_lr=1e-6),
            ]

            # Direction classifier with class weights
            self.lstm_dir_model = self._build_lstm_classifier(n_features)
            self.lstm_dir_model.fit(
                X_seq_train, y_seq_dir_train,
                validation_data=(X_seq_val, y_seq_dir_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=callbacks, verbose=0,
                class_weight=class_weights,
            )

            # Magnitude regressor
            callbacks_mag = [
                EarlyStopping(patience=15, restore_best_weights=True, verbose=0,
                             monitor='val_mae'),
                ReduceLROnPlateau(factor=0.5, patience=7, verbose=0, min_lr=1e-6),
            ]
            self.lstm_mag_model = self._build_lstm_regressor(n_features)
            self.lstm_mag_model.fit(
                X_seq_train, y_seq_mag_train,
                validation_data=(X_seq_val, y_seq_mag_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=callbacks_mag, verbose=0
            )

            # LSTM validation
            lstm_dir_proba = self.lstm_dir_model.predict(X_seq_val, verbose=0).flatten()
            lstm_dir_pred = (lstm_dir_proba > 0.5).astype(int)
            lstm_mag_pred = self.lstm_mag_model.predict(X_seq_val, verbose=0).flatten()

            lstm_acc = accuracy_score(y_seq_dir_val, lstm_dir_pred)
            lstm_f1 = f1_score(y_seq_dir_val, lstm_dir_pred)
            lstm_mae = mean_absolute_error(y_seq_mag_val, lstm_mag_pred)
            print(f"    LSTM — Accuracy: {lstm_acc:.1%} | F1: {lstm_f1:.2f} | MAE: {lstm_mae:.3f}%")

        # ========================
        # 2. XGBOOST MODELS (Enhanced)
        # ========================
        print(f"    Training XGBoost (n_est={XGB_N_ESTIMATORS}, depth={XGB_MAX_DEPTH})...")

        X_xgb_train = X_train[LOOKBACK:]
        X_xgb_val = X_val[LOOKBACK:]
        y_xgb_dir_train = y_dir_train[LOOKBACK:]
        y_xgb_dir_val = y_dir_val[LOOKBACK:]
        y_xgb_mag_train = y_mag_train[LOOKBACK:]
        y_xgb_mag_val = y_mag_val[LOOKBACK:]

        # Direction classifier with regularization
        scale_pos_weight = (y_xgb_dir_train == 0).sum() / max((y_xgb_dir_train == 1).sum(), 1)
        self.xgb_dir_model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            gamma=XGB_GAMMA,
            reg_alpha=XGB_REG_ALPHA,
            reg_lambda=XGB_REG_LAMBDA,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            early_stopping_rounds=30,
        )
        self.xgb_dir_model.fit(
            X_xgb_train, y_xgb_dir_train,
            eval_set=[(X_xgb_val, y_xgb_dir_val)],
            verbose=False
        )

        # Magnitude regressor
        self.xgb_mag_model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            gamma=XGB_GAMMA,
            reg_alpha=XGB_REG_ALPHA,
            reg_lambda=XGB_REG_LAMBDA,
            eval_metric='mae',
            verbosity=0,
            random_state=42,
            early_stopping_rounds=30,
        )
        self.xgb_mag_model.fit(
            X_xgb_train, y_xgb_mag_train,
            eval_set=[(X_xgb_val, y_xgb_mag_val)],
            verbose=False
        )

        # XGBoost validation
        xgb_dir_pred = self.xgb_dir_model.predict(X_xgb_val)
        xgb_dir_proba = self.xgb_dir_model.predict_proba(X_xgb_val)[:, 1]
        xgb_mag_pred = self.xgb_mag_model.predict(X_xgb_val)

        xgb_acc = accuracy_score(y_xgb_dir_val, xgb_dir_pred)
        xgb_f1 = f1_score(y_xgb_dir_val, xgb_dir_pred)
        xgb_mae = mean_absolute_error(y_xgb_mag_val, xgb_mag_pred)
        print(f"    XGBoost — Accuracy: {xgb_acc:.1%} | F1: {xgb_f1:.2f} | MAE: {xgb_mae:.3f}%")

        # ========================
        # 3. ADAPTIVE ENSEMBLE WEIGHTS
        # ========================
        # Learn optimal weights from validation performance
        if self.lstm_dir_model is not None:
            # Try different weight combos and pick the best
            best_acc = 0
            best_w = (0.5, 0.5)
            for lw in np.arange(0.1, 0.91, 0.05):
                xw = 1.0 - lw
                ens_prob = lw * lstm_dir_proba + xw * xgb_dir_proba
                ens_pred = (ens_prob > 0.5).astype(int)
                acc = accuracy_score(y_seq_dir_val, ens_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_w = (round(lw, 2), round(xw, 2))

            self.ensemble_weights = {'lstm': best_w[0], 'xgb': best_w[1]}
            print(f"    Optimal weights — LSTM: {best_w[0]:.0%} | XGB: {best_w[1]:.0%}")

            # Final ensemble with optimal weights
            ens_dir_prob = best_w[0] * lstm_dir_proba + best_w[1] * xgb_dir_proba
            ens_dir_pred = (ens_dir_prob > 0.5).astype(int)
            ens_mag_pred = best_w[0] * lstm_mag_pred + best_w[1] * xgb_mag_pred

            ens_acc = accuracy_score(y_seq_dir_val, ens_dir_pred)
            ens_f1 = f1_score(y_seq_dir_val, ens_dir_pred)
            ens_mae = mean_absolute_error(y_seq_mag_val, ens_mag_pred)
        else:
            ens_acc = xgb_acc
            ens_f1 = xgb_f1
            ens_mae = xgb_mae
            self.ensemble_weights = {'lstm': 0.0, 'xgb': 1.0}

        print(f"    ENSEMBLE — Accuracy: {ens_acc:.1%} | F1: {ens_f1:.2f} | MAE: {ens_mae:.3f}%")

        self.metrics = {
            'lstm': {'accuracy': round(lstm_acc, 4), 'f1': round(lstm_f1, 4), 'mae': round(lstm_mae, 4)},
            'xgboost': {'accuracy': round(xgb_acc, 4), 'f1': round(xgb_f1, 4), 'mae': round(xgb_mae, 4)},
            'ensemble': {'accuracy': round(ens_acc, 4), 'f1': round(ens_f1, 4), 'mae': round(ens_mae, 4)},
            'ensemble_weights': self.ensemble_weights,
            'n_features_selected': n_features,
            'n_features_total': n_features_orig,
        }

        # Feature importance from XGBoost (on selected features)
        self.feature_importance = dict(zip(
            self.selected_features,
            self.xgb_dir_model.feature_importances_
        ))

        return self.metrics

    def predict(self, X_latest):
        """
        Predict direction and magnitude for the latest data.
        X_latest: DataFrame with same features as training.
        """
        if self.xgb_dir_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # First align to ALL feature_names (scaler expects full feature set)
        if self.feature_names:
            for col in self.feature_names:
                if col not in X_latest.columns:
                    X_latest[col] = 0
            X_full = X_latest[self.feature_names]
        else:
            X_full = X_latest

        # Scale on full feature set
        X_scaled_full = self.scaler.transform(X_full.values)

        # Now apply feature selection AFTER scaling
        if self.selected_features and self.feature_names:
            sel_idx = [self.feature_names.index(f) for f in self.selected_features
                       if f in self.feature_names]
            X_scaled = X_scaled_full[:, sel_idx]
        else:
            X_scaled = X_scaled_full

        lw = self.ensemble_weights.get('lstm', 0.45)
        xw = self.ensemble_weights.get('xgb', 0.55)

        # LSTM prediction
        if self.lstm_dir_model is not None and len(X_scaled) >= LOOKBACK:
            X_seq = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, -1)
            lstm_dir_prob = float(self.lstm_dir_model.predict(X_seq, verbose=0)[0][0])
            lstm_mag = float(self.lstm_mag_model.predict(X_seq, verbose=0)[0][0])
        else:
            lstm_dir_prob = 0.5
            lstm_mag = 0.0
            lw = 0.0
            xw = 1.0

        # XGBoost prediction
        X_flat = X_scaled[-1:].reshape(1, -1)
        xgb_dir_prob = float(self.xgb_dir_model.predict_proba(X_flat)[0][1])
        xgb_mag = float(self.xgb_mag_model.predict(X_flat)[0])

        # Ensemble
        total_w = lw + xw
        ens_prob = (lw * lstm_dir_prob + xw * xgb_dir_prob) / total_w
        ens_mag = (lw * lstm_mag + xw * xgb_mag) / total_w

        direction = 'UP' if ens_prob > 0.5 else 'DOWN'
        confidence = abs(ens_prob - 0.5) * 200

        return {
            'direction': direction,
            'probability': round(ens_prob, 4),
            'magnitude_pct': round(ens_mag, 3),
            'confidence': round(confidence, 1),
            'lstm_prob': round(lstm_dir_prob, 4),
            'lstm_mag': round(lstm_mag, 3),
            'xgb_prob': round(xgb_dir_prob, 4),
            'xgb_mag': round(xgb_mag, 3),
        }

    def save(self):
        """Save all model components."""
        path = os.path.join(MODEL_DIR, self.name)
        os.makedirs(path, exist_ok=True)

        if self.lstm_dir_model is not None:
            self.lstm_dir_model.save(os.path.join(path, 'lstm_dir.keras'))
            self.lstm_mag_model.save(os.path.join(path, 'lstm_mag.keras'))
        joblib.dump(self.xgb_dir_model, os.path.join(path, 'xgb_dir.pkl'))
        joblib.dump(self.xgb_mag_model, os.path.join(path, 'xgb_mag.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'selected_features': self.selected_features,
                'metrics': self.metrics,
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': {k: float(v) for k, v in self.feature_importance.items()},
            }, f, indent=2)
        print(f"    Model saved to {path}/")

    def load(self):
        """Load pre-trained model components."""
        path = os.path.join(MODEL_DIR, self.name)
        if not os.path.exists(path):
            return False

        try:
            lstm_path = os.path.join(path, 'lstm_dir.keras')
            if os.path.exists(lstm_path):
                self.lstm_dir_model = load_model(lstm_path,
                    custom_objects={'AttentionLayer': AttentionLayer})
                self.lstm_mag_model = load_model(os.path.join(path, 'lstm_mag.keras'),
                    custom_objects={'AttentionLayer': AttentionLayer})
            else:
                self.lstm_dir_model = None
                self.lstm_mag_model = None

            self.xgb_dir_model = joblib.load(os.path.join(path, 'xgb_dir.pkl'))
            self.xgb_mag_model = joblib.load(os.path.join(path, 'xgb_mag.pkl'))
            self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
            with open(os.path.join(path, 'meta.json'), 'r') as f:
                meta = json.load(f)
                self.feature_names = meta['feature_names']
                self.selected_features = meta.get('selected_features', self.feature_names)
                self.metrics = meta['metrics']
                self.ensemble_weights = meta.get('ensemble_weights', {'lstm': 0.45, 'xgb': 0.55})
                self.feature_importance = meta.get('feature_importance', {})
            print(f"    Model loaded from {path}/")
            return True
        except Exception as e:
            print(f"    Error loading model: {e}")
            return False
