"""
Enhanced ML Model Training with Versioning, Metrics, and Logging
Supports model versioning, comprehensive metrics, training logs, and data validation
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, precision_score, recall_score, f1_score
)
import joblib
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Import database config loader
from train_model import load_db_config, get_db_connection

DB_CONFIG = load_db_config()

def validate_training_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate training data before training
    Returns: (is_valid, error_messages)
    """
    errors = []
    
    if df is None or df.empty:
        errors.append("Dataset is empty or None")
        return False, errors
    
    # Check required columns
    required_cols = ['current_average', 'attendance_rate', 'assessments_completed', 'total_score', 'total_max']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check minimum data size
    if len(df) < 10:
        errors.append(f"Insufficient data: {len(df)} samples (minimum 10 required)")
    
    # Check for null values in critical columns
    for col in required_cols:
        if col in df.columns and df[col].isnull().sum() > len(df) * 0.5:
            errors.append(f"Too many null values in {col} (>50%)")
    
    # Check data ranges
    if 'current_average' in df.columns:
        if df['current_average'].min() < 0 or df['current_average'].max() > 100:
            errors.append("current_average should be between 0 and 100")
    
    if 'attendance_rate' in df.columns:
        if df['attendance_rate'].min() < 0 or df['attendance_rate'].max() > 100:
            errors.append("attendance_rate should be between 0 and 100")
    
    return len(errors) == 0, errors

def get_next_model_version(conn) -> str:
    """Get next model version number"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(CAST(SUBSTRING_INDEX(model_version, '.', 1) AS UNSIGNED)) as max_major FROM ml_models")
        result = cursor.fetchone()
        major = (result[0] or 0) + 1
        cursor.close()
        return f"{major}.0"
    except Exception as e:
        print(f"Error getting model version: {e}")
        return "1.0"

def create_training_log(conn, data_source: str, triggered_by_user_id: Optional[int] = None) -> int:
    """Create a new training log entry and return its ID"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ml_training_logs 
            (training_status, data_source, triggered_by_user_id, created_at)
            VALUES ('processing', %s, %s, NOW())
        """, (data_source, triggered_by_user_id))
        log_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        return log_id
    except Exception as e:
        print(f"Error creating training log: {e}")
        return 0

def update_training_log(conn, log_id: int, updates: Dict):
    """Update training log with metrics and status"""
    try:
        cursor = conn.cursor()
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = %s")
            values.append(value)
        
        if set_clauses:
            query = f"UPDATE ml_training_logs SET {', '.join(set_clauses)} WHERE training_log_id = %s"
            values.append(log_id)
            cursor.execute(query, values)
            conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error updating training log: {e}")

def save_model_version(conn, model_version: str, model_type: str, file_path: str, training_log_id: int, is_active: bool = True):
    """Save model version to database"""
    try:
        cursor = conn.cursor()
        
        # Deactivate previous models if activating new one
        if is_active:
            cursor.execute("UPDATE ml_models SET is_active = 0 WHERE model_type IN ('both', %s)", (model_type,))
        
        # Insert new model version
        cursor.execute("""
            INSERT INTO ml_models 
            (model_version, model_type, file_path, is_active, training_log_id, created_at, activated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
            is_active = VALUES(is_active),
            training_log_id = VALUES(training_log_id),
            activated_at = NOW()
        """, (model_version, model_type, file_path, 1 if is_active else 0, training_log_id))
        
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error saving model version: {e}")

def calculate_comprehensive_metrics(y_true, y_pred, model_type: str = 'classifier'):
    """Calculate comprehensive metrics for model evaluation"""
    metrics = {}
    
    if model_type == 'classifier':
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(list(set(y_true) | set(y_pred)))
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'classes': classes,
            'labels': {i: classes[i] for i in range(len(classes))}
        }
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
    elif model_type == 'regressor':
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['r2_score'] = float(r2_score(y_true, y_pred))
        metrics['mean_error'] = float(np.mean(np.abs(y_true - y_pred)))
        
        # Additional statistics
        metrics['max_error'] = float(np.max(np.abs(y_true - y_pred)))
        metrics['median_error'] = float(np.median(np.abs(y_true - y_pred)))
    
    return metrics

def train_models_enhanced(
    df: Optional[pd.DataFrame] = None,
    data_source: str = 'real',
    triggered_by_user_id: Optional[int] = None,
    use_uploaded_data: bool = False
) -> Dict:
    """
    Enhanced training function with versioning, metrics, and logging
    
    Returns:
        Dictionary with training results, metrics, and model version
    """
    start_time = time.time()
    conn = get_db_connection()
    training_log_id = 0
    model_version = "1.0"
    
    try:
        # Create training log
        if conn:
            training_log_id = create_training_log(conn, data_source, triggered_by_user_id)
            model_version = get_next_model_version(conn)
        
        # Fetch data if not provided
        if df is None:
            if data_source == 'real' and not use_uploaded_data:
                from train_model import fetch_training_data
                df = fetch_training_data()
            else:
                df = None
        
        # Validate data
        if df is None or df.empty:
            if conn and training_log_id:
                update_training_log(conn, training_log_id, {
                    'training_status': 'failed',
                    'error_message': 'No training data available',
                    'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            return {
                'success': False,
                'error': 'No training data available',
                'training_log_id': training_log_id
            }
        
        is_valid, errors = validate_training_data(df)
        if not is_valid:
            if conn and training_log_id:
                update_training_log(conn, training_log_id, {
                    'training_status': 'failed',
                    'error_message': '; '.join(errors),
                    'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            return {
                'success': False,
                'error': 'Data validation failed',
                'errors': errors,
                'training_log_id': training_log_id
            }
        
        # Update log with dataset size
        dataset_size = len(df)
        if conn and training_log_id:
            update_training_log(conn, training_log_id, {
                'dataset_size': dataset_size
            })
        
        # Prepare features
        from train_model import prepare_features
        X_class, features = prepare_features(df[df['risk_level'].notna()].copy() if 'risk_level' in df.columns else df.copy())
        X_reg, _ = prepare_features(df[df['final_grade'].notna()].copy() if 'final_grade' in df.columns else df.copy())
        
        # Prepare targets
        df_classification = df[df['risk_level'].notna()].copy() if 'risk_level' in df.columns else pd.DataFrame()
        df_regression = df[df['final_grade'].notna()].copy() if 'final_grade' in df.columns else pd.DataFrame()
        
        y_risk = df_classification['risk_level'].values if not df_classification.empty else None
        y_grade = df_regression['final_grade'].values if not df_regression.empty else None
        
        results = {
            'success': True,
            'model_version': model_version,
            'dataset_size': dataset_size,
            'training_log_id': training_log_id,
            'classifier': {},
            'regressor': {}
        }
        
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        scaler = StandardScaler()
        
        # Train Classifier
        if y_risk is not None and len(y_risk) > 10:
            print(f"\n📊 Training Risk Classification Model (v{model_version})...")
            
            X_class_scaled = scaler.fit_transform(X_class)
            
            # Split data
            if len(X_class) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_class_scaled, y_risk, test_size=0.2, random_state=42, stratify=y_risk
                )
            else:
                X_train, X_test, y_train, y_test = X_class_scaled, X_class_scaled, y_risk, y_risk
            
            classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test)
            metrics = calculate_comprehensive_metrics(y_test, y_pred, 'classifier')
            
            # Save model
            classifier_path = f'{model_dir}/classifier_v{model_version}.pkl'
            joblib.dump(classifier, classifier_path)
            
            # Save version to database
            if conn:
                save_model_version(conn, model_version, 'classifier', classifier_path, training_log_id, True)
                update_training_log(conn, training_log_id, {
                    'training_samples': len(X_train),
                    'validation_samples': len(X_test),
                    'classifier_accuracy': metrics['accuracy'],
                    'classifier_precision': metrics['precision'],
                    'classifier_recall': metrics['recall'],
                    'classifier_f1_score': metrics['f1_score'],
                    'classifier_confusion_matrix': json.dumps(metrics['confusion_matrix'])
                })
            
            results['classifier'] = {
                'trained': True,
                'metrics': metrics,
                'file_path': classifier_path
            }
            
            print(f"   ✅ Accuracy: {metrics['accuracy']:.2%}")
            print(f"   ✅ Precision: {metrics['precision']:.2%}")
            print(f"   ✅ Recall: {metrics['recall']:.2%}")
            print(f"   ✅ F1 Score: {metrics['f1_score']:.2%}")
        else:
            results['classifier'] = {'trained': False, 'reason': 'Insufficient data'}
        
        # Train Regressor
        if y_grade is not None and len(y_grade) > 10:
            print(f"\n📊 Training Grade Prediction Model (v{model_version})...")
            
            X_reg_scaled = scaler.fit_transform(X_reg)
            
            # Split data
            if len(X_reg) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_reg_scaled, y_grade, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X_reg_scaled, X_reg_scaled, y_grade, y_grade
            
            regressor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            regressor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = regressor.predict(X_test)
            metrics = calculate_comprehensive_metrics(y_test, y_pred, 'regressor')
            
            # Save model
            regressor_path = f'{model_dir}/regressor_v{model_version}.pkl'
            joblib.dump(regressor, regressor_path)
            
            # Save scaler
            scaler_path = f'{model_dir}/scaler_v{model_version}.pkl'
            joblib.dump(scaler, scaler_path)
            
            # Save version to database
            if conn:
                save_model_version(conn, model_version, 'regressor', regressor_path, training_log_id, True)
                update_training_log(conn, training_log_id, {
                    'regressor_mae': metrics['mae'],
                    'regressor_rmse': metrics['rmse'],
                    'regressor_r2_score': metrics['r2_score'],
                    'regressor_mean_error': metrics['mean_error']
                })
            
            results['regressor'] = {
                'trained': True,
                'metrics': metrics,
                'file_path': regressor_path
            }
            
            print(f"   ✅ MAE: {metrics['mae']:.2f}")
            print(f"   ✅ RMSE: {metrics['rmse']:.2f}")
            print(f"   ✅ R² Score: {metrics['r2_score']:.4f}")
        else:
            results['regressor'] = {'trained': False, 'reason': 'Insufficient data'}
        
        # Update training log as completed
        training_duration = int(time.time() - start_time)
        if conn and training_log_id:
            update_training_log(conn, training_log_id, {
                'training_status': 'completed',
                'model_version': model_version,
                'training_duration_seconds': training_duration,
                'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        results['training_duration_seconds'] = training_duration
        print(f"\n✅ Training completed in {training_duration} seconds")
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Training failed: {error_msg}")
        
        if conn and training_log_id:
            update_training_log(conn, training_log_id, {
                'training_status': 'failed',
                'error_message': error_msg,
                'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return {
            'success': False,
            'error': error_msg,
            'training_log_id': training_log_id
        }
    finally:
        if conn:
            conn.close()

