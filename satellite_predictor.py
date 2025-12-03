"""
Satellite Error Prediction Model
ISRO SIH 2025 - Problem Statement 25176
Predicts satellite ephemeris and clock errors using LSTM neural networks
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')


class SatelliteErrorPredictor:
    """
    A class to predict satellite ephemeris and clock errors using LSTM models.
    
    This model predicts the 8th day errors based on 7 days of historical data.
    """
    
    def __init__(self, sequence_length=7):
        """
        Initialize the predictor.
        
        Args:
            sequence_length (int): Number of time steps to look back (default: 7 days)
        """
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
        
    def load_and_preprocess_data(self, file_path, satellite_type='MEO'):
        """
        Load and preprocess satellite data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            satellite_type (str): Type of satellite orbit ('GEO' or 'MEO')
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print(f"\n{'='*60}")
        print(f"Loading {satellite_type} data from: {file_path}")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        column_mapping = {
            'x_error  (m)': 'x_error (m)',
            'y_error  (m)': 'y_error (m)',
            'z_error  (m)': 'z_error (m)',
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert time to datetime
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        df = df.sort_values('utc_time').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} records")
        print(f"✓ Date range: {df['utc_time'].min()} to {df['utc_time'].max()}")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"⚠ Found {missing_before} missing values - filling with interpolation")
            df[self.feature_columns] = df[self.feature_columns].interpolate(method='linear')
            df.fillna(method='bfill', inplace=True)
        
        # Outlier detection and treatment using IQR method
        print("\n🔍 Detecting and treating outliers...")
        outliers_treated = 0
        
        for col in self.feature_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive treatment
            upper_bound = Q3 + 3 * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                print(f"  • {col}: {outliers_count} outliers detected")
                # Cap outliers instead of removing
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                outliers_treated += outliers_count
        
        print(f"✓ Treated {outliers_treated} outliers (capped to acceptable range)")
        
        # Feature engineering
        print("\n🔧 Engineering features...")
        
        # Total position error (3D Euclidean distance)
        df['total_position_error'] = np.sqrt(
            df['x_error (m)']**2 + 
            df['y_error (m)']**2 + 
            df['z_error (m)']**2
        )
        
        # Extract time features
        df['hour'] = df['utc_time'].dt.hour
        df['day'] = df['utc_time'].dt.day
        df['day_of_week'] = df['utc_time'].dt.dayofweek
        
        # Rolling statistics (if enough data)
        if len(df) > 10:
            for col in self.feature_columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        
        print(f"✓ Created {len(df.columns) - len(self.feature_columns) - 1} additional features")
        
        return df
    
    def create_sequences(self, data, target_col):
        """
        Create sequences for time series prediction.
        
        Args:
            data (np.array): Input data
            target_col (int): Index of target column
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_col])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM neural network model.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df, satellite_type='MEO', epochs=50, batch_size=32):
        """
        Train LSTM models for each error component.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            satellite_type (str): Type of satellite
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history for each component
        """
        print(f"\n{'='*60}")
        print(f"Training Models for {satellite_type}")
        print(f"{'='*60}")
        
        histories = {}
        
        # Select features for training
        feature_cols = [col for col in df.columns if col not in ['utc_time']]
        data = df[feature_cols].values
        
        # Train a model for each error component
        for idx, target in enumerate(self.feature_columns):
            print(f"\n📊 Training model for: {target}")
            print(f"{'-'*60}")
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers[f'{satellite_type}_{target}'] = scaler
            
            # Create sequences
            target_idx = feature_cols.index(target)
            X, y = self.create_sequences(scaled_data, target_idx)
            
            if len(X) < 10:
                print(f"⚠ Warning: Not enough data to train {target} model")
                continue
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            print(f"  • Training samples: {len(X_train)}")
            print(f"  • Validation samples: {len(X_val)}")
            print(f"  • Input shape: {X_train.shape}")
            
            # Build and train model
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.models[f'{satellite_type}_{target}'] = model
            histories[target] = history.history
            
            # Evaluate
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            print(f"  ✓ Training complete!")
            print(f"    - Final training loss: {train_loss:.6f}")
            print(f"    - Final validation loss: {val_loss:.6f}")
        
        return histories
    
    def predict_8th_day(self, df, satellite_type='MEO'):
        """
        Predict errors for the 8th day based on last 7 days of data.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            satellite_type (str): Type of satellite
            
        Returns:
            dict: Predictions for each error component
        """
        print(f"\n{'='*60}")
        print(f"Predicting 8th Day Errors for {satellite_type}")
        print(f"{'='*60}")
        
        predictions = {}
        
        feature_cols = [col for col in df.columns if col not in ['utc_time']]
        
        for target in self.feature_columns:
            model_key = f'{satellite_type}_{target}'
            
            if model_key not in self.models:
                print(f"⚠ Warning: No model found for {target}")
                continue
            
            # Get last sequence
            data = df[feature_cols].values
            scaler = self.scalers[model_key]
            scaled_data = scaler.transform(data)
            
            # Take last sequence_length time steps
            last_sequence = scaled_data[-self.sequence_length:]
            last_sequence = last_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict
            model = self.models[model_key]
            scaled_prediction = model.predict(last_sequence, verbose=0)
            
            # Inverse transform to get actual value
            dummy = np.zeros((1, len(feature_cols)))
            target_idx = feature_cols.index(target)
            dummy[0, target_idx] = scaled_prediction[0, 0]
            prediction = scaler.inverse_transform(dummy)[0, target_idx]
            
            predictions[target] = prediction
            print(f"  • {target}: {prediction:.6f} m")
        
        return predictions
    
    def evaluate_predictions(self, predictions, ground_truth):
        """
        Evaluate predictions using Shapiro-Wilk test for normality.
        
        Args:
            predictions (np.array): Predicted values
            ground_truth (np.array): Actual values
            
        Returns:
            dict: Evaluation metrics
        """
        residuals = ground_truth - predictions
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # Other metrics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        results = {
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05,  # If p > 0.05, residuals are normally distributed
            'rmse': rmse,
            'mae': mae,
            'residuals': residuals
        }
        
        return results
    
    def save_models(self, save_dir='models'):
        """
        Save trained models and scalers.
        
        Args:
            save_dir (str): Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model.save(f'{save_dir}/{name}_model.h5')
        
        with open(f'{save_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print(f"\n✓ Models saved to {save_dir}/")
    
    def load_models(self, save_dir='models'):
        """
        Load trained models and scalers.
        
        Args:
            save_dir (str): Directory containing saved models
        """
        import os
        
        # Load models
        for file in os.listdir(save_dir):
            if file.endswith('_model.h5'):
                name = file.replace('_model.h5', '')
                self.models[name] = keras.models.load_model(f'{save_dir}/{file}')
        
        # Load scalers
        with open(f'{save_dir}/scalers.pkl', 'rb') as f:
            self.scalers = pickle.load(f)
        
        print(f"✓ Models loaded from {save_dir}/")


def main():
    """
    Main function to demonstrate the satellite error prediction system.
    """
    print("\n" + "="*60)
    print("SATELLITE ERROR PREDICTION SYSTEM")
    print("ISRO SIH 2025 - Problem Statement 25176")
    print("="*60)
    
    # Initialize predictor
    predictor = SatelliteErrorPredictor(sequence_length=7)
    
    # Load and process data
    geo_df = predictor.load_and_preprocess_data('DATA_GEO_Train.csv', 'GEO')
    meo_df = predictor.load_and_preprocess_data('DATA_MEO_Train.csv', 'MEO')
    
    # Train models
    geo_history = predictor.train_model(geo_df, 'GEO', epochs=50, batch_size=16)
    meo_history = predictor.train_model(meo_df, 'MEO', epochs=50, batch_size=32)
    
    # Make predictions for 8th day
    geo_predictions = predictor.predict_8th_day(geo_df, 'GEO')
    meo_predictions = predictor.predict_8th_day(meo_df, 'MEO')
    
    # Save models
    predictor.save_models('models')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nUse the Streamlit app to visualize results and make predictions.")
    print("Run: streamlit run app.py")
    

if __name__ == "__main__":
    main()
