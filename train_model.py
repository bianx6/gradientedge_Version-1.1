"""
Train ML Models with Real Database Data
Fetches actual student performance data from MySQL database and trains models
"""

import sys
import os
import json

# Add parent directory to path to import database config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    print("Warning: mysql-connector-python not installed. Install with: pip install mysql-connector-python")
    MYSQL_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Database configuration - Load from PHP config file
def load_db_config():
    """Load database configuration from PHP config file"""
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'database.php')
    
    # Default config
    config = {
        'host': 'localhost',
        'database': 'u209814635_gradientdb',
        'user': 'u209814635_gradientedge25',
        'password': '25Gradient@edge',
        'port': 3306
    }
    
    # Try to parse PHP config file
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            # Extract database config (simple parsing)
            import re
            host_match = re.search(r"DB_HOST.*?['\"](.*?)['\"]", content)
            db_match = re.search(r"DB_NAME.*?['\"](.*?)['\"]", content)
            user_match = re.search(r"DB_USER.*?['\"](.*?)['\"]", content)
            pass_match = re.search(r"DB_PASS.*?['\"](.*?)['\"]", content)
            port_match = re.search(r"DB_PORT.*?(\d+)", content)
            
            if host_match:
                config['host'] = host_match.group(1)
            if db_match:
                config['database'] = db_match.group(1)
            if user_match:
                config['user'] = user_match.group(1)
            if pass_match:
                config['password'] = pass_match.group(1)
            if port_match:
                config['port'] = int(port_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse PHP config: {e}")
        print("Using default database configuration")
    
    return config

DB_CONFIG = load_db_config()

def get_db_connection():
    """Create database connection"""
    if not MYSQL_AVAILABLE:
        print("MySQL connector not available")
        return None
    
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_training_data():
    """Fetch real training data from database"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database. Using synthetic data.")
        return None
    
    try:
        query = """
        SELECT 
            -- Student performance features
            COALESCE(
                (SELECT AVG((ass.score / ass2.total_points) * 100)
                 FROM assessment_scores ass
                 JOIN assessments ass2 ON ass.assessment_id = ass2.assessment_id
                 WHERE ass.student_id = e.student_id 
                 AND ass2.class_id = e.class_id
                 AND ass2.status = 'active'), 0
            ) as current_average,
            
            -- Attendance rate
            COALESCE(
                (SELECT (SUM(CASE WHEN ar.status IN ('present', 'late', 'excused') THEN 1 ELSE 0 END) / 
                         NULLIF(COUNT(*), 0)) * 100
                 FROM attendance_records ar
                 WHERE ar.student_id = e.student_id 
                 AND ar.class_id = e.class_id), 100
            ) as attendance_rate,
            
            -- Assessments completed
            COALESCE(
                (SELECT COUNT(DISTINCT ass.assessment_id)
                 FROM assessment_scores ass
                 JOIN assessments a ON ass.assessment_id = a.assessment_id
                 WHERE ass.student_id = e.student_id 
                 AND a.class_id = e.class_id
                 AND a.status = 'active'), 0
            ) as assessments_completed,
            
            -- Total score
            COALESCE(
                (SELECT SUM(ass.score)
                 FROM assessment_scores ass
                 JOIN assessments a ON ass.assessment_id = a.assessment_id
                 WHERE ass.student_id = e.student_id 
                 AND a.class_id = e.class_id
                 AND a.status = 'active'), 0
            ) as total_score,
            
            -- Total max points
            COALESCE(
                (SELECT SUM(a.total_points)
                 FROM assessments a
                 WHERE a.class_id = e.class_id
                 AND a.status = 'active'), 0
            ) as total_max,
            
            -- Actual final grade (target for regression): use IG% (final_grades.initial_grade)
            COALESCE(fg.initial_grade, NULL) as final_grade,
            
            -- Risk level (derived from final grade)
            CASE 
                WHEN fg.initial_grade IS NULL THEN NULL
                WHEN fg.initial_grade < 60 THEN 'High'
                WHEN fg.initial_grade < 75 THEN 'Medium'
                ELSE 'Low'
            END as risk_level
            
        FROM enrollments e
        LEFT JOIN final_grades fg ON e.student_id = fg.student_id AND e.class_id = fg.class_id
        WHERE e.status = 'enrolled'
        AND EXISTS (
            SELECT 1 FROM assessment_scores ass
            JOIN assessments a ON ass.assessment_id = a.assessment_id
            WHERE ass.student_id = e.student_id 
            AND a.class_id = e.class_id
            AND a.status = 'active'
        )
        HAVING current_average > 0 OR assessments_completed > 0
        LIMIT 5000
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            print("No training data found in database. Using synthetic data.")
            return None
        
        print(f"Fetched {len(df)} training samples from database")
        return df
        
    except Error as e:
        print(f"Error fetching training data: {e}")
        if conn:
            conn.close()
        return None

def prepare_features(df):
    """Prepare feature matrix from dataframe"""
    # Calculate derived features
    df['completion_rate'] = df['assessments_completed'] / 30.0  # Normalize
    df['completion_rate'] = df['completion_rate'].clip(0, 1)
    
    df['score_ratio'] = df.apply(
        lambda row: row['total_score'] / row['total_max'] if row['total_max'] > 0 else 0,
        axis=1
    )
    df['score_ratio'] = df['score_ratio'].clip(0, 1)
    
    # Create feature matrix
    features = ['current_average', 'attendance_rate', 'completion_rate', 'score_ratio', 'assessments_completed']
    X = df[features].fillna(0).values
    
    return X, features

def train_models_with_real_data():
    """Train models using real database data"""
    print("=" * 60)
    print("Training ML Models with Real Database Data")
    print("=" * 60)
    
    # Fetch training data
    df = fetch_training_data()
    
    if df is None or df.empty:
        print("\n⚠️  No real data available. Using synthetic data for training.")
        return train_models_synthetic()
    
    # Filter out rows with missing target values
    df_classification = df[df['risk_level'].notna()].copy()
    df_regression = df[df['final_grade'].notna()].copy()
    
    if df_classification.empty and df_regression.empty:
        print("\n⚠️  No complete data available. Using synthetic data for training.")
        return train_models_synthetic()
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare features
    X_class, features = prepare_features(df_classification)
    X_reg, _ = prepare_features(df_regression)
    
    # Prepare targets
    y_risk = df_classification['risk_level'].values if not df_classification.empty else None
    y_grade = df_regression['final_grade'].values if not df_regression.empty else None
    
    # Scale features
    scaler = StandardScaler()
    
    results = {}
    
    # Train Classifier (Risk Level Prediction)
    if y_risk is not None and len(y_risk) > 10:
        print(f"\n📊 Training Risk Classification Model...")
        print(f"   Training samples: {len(X_class)}")
        
        X_class_scaled = scaler.fit_transform(X_class)
        
        # Split data
        if len(X_class) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_class_scaled, y_risk, test_size=0.2, random_state=42, stratify=y_risk
            )
        else:
            X_train, X_test, y_train, y_test = X_class_scaled, X_class_scaled, y_risk, y_risk
        
        # Train classifier
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results['classifier_accuracy'] = accuracy
        
        print(f"   ✅ Risk Classification Accuracy: {accuracy:.2%}")
        print(f"   📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save classifier
        joblib.dump(classifier, f'{model_dir}/classifier.pkl')
        print(f"   💾 Classifier saved to {model_dir}/classifier.pkl")
    else:
        print(f"\n⚠️  Insufficient data for classification ({len(X_class) if y_risk is not None else 0} samples). Using synthetic data.")
        classifier, _, _ = train_models_synthetic()
        results['classifier_accuracy'] = 'N/A (synthetic)'
    
    # Train Regressor (Final Grade Prediction)
    if y_grade is not None and len(y_grade) > 10:
        print(f"\n📊 Training Grade Prediction Model...")
        print(f"   Training samples: {len(X_reg)}")
        
        X_reg_scaled = scaler.fit_transform(X_reg)
        
        # Split data
        if len(X_reg) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_reg_scaled, y_grade, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X_reg_scaled, X_reg_scaled, y_grade, y_grade
        
        # Train regressor
        regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        regressor.fit(X_train, y_train)
        
        # Evaluate
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        results['regressor_rmse'] = rmse
        
        print(f"   ✅ Grade Prediction RMSE: {rmse:.2f}")
        print(f"   📊 Mean Error: {np.mean(np.abs(y_test - y_pred)):.2f} points")
        
        # Save regressor
        joblib.dump(regressor, f'{model_dir}/regressor.pkl')
        print(f"   💾 Regressor saved to {model_dir}/regressor.pkl")
    else:
        print(f"\n⚠️  Insufficient data for regression ({len(X_reg) if y_grade is not None else 0} samples). Using synthetic data.")
        _, regressor, _ = train_models_synthetic()
        results['regressor_rmse'] = 'N/A (synthetic)'
    
    # Save scaler
    if y_risk is not None and len(y_risk) > 10:
        joblib.dump(scaler, f'{model_dir}/scaler.pkl')
        print(f"   💾 Scaler saved to {model_dir}/scaler.pkl")
    elif y_grade is not None and len(y_grade) > 10:
        # Fit scaler on regression data if classification data not available
        X_reg_scaled = scaler.fit_transform(X_reg)
        joblib.dump(scaler, f'{model_dir}/scaler.pkl')
        print(f"   💾 Scaler saved to {model_dir}/scaler.pkl")
    
    print("\n" + "=" * 60)
    print("✅ Model Training Complete!")
    print("=" * 60)
    print(f"Results: {json.dumps(results, indent=2)}")
    
    return classifier, regressor, scaler

def train_models_synthetic():
    """Train models with synthetic data (fallback)"""
    print("\n📊 Training with synthetic data...")
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    current_avg = np.random.uniform(50, 100, n_samples)
    attendance_rate = np.random.uniform(60, 100, n_samples)
    assessments_completed = np.random.randint(5, 30, n_samples)
    total_score = np.random.uniform(200, 800, n_samples)
    total_max = np.random.uniform(400, 1000, n_samples)
    
    # Calculate derived features
    completion_rate = assessments_completed / 30.0
    score_ratio = total_score / total_max
    
    # Create feature matrix
    X = np.column_stack([
        current_avg,
        attendance_rate,
        completion_rate,
        score_ratio,
        assessments_completed
    ])
    
    # Generate risk labels
    risk_score = (
        (100 - current_avg) * 0.4 +
        (100 - attendance_rate) * 0.3 +
        (1 - completion_rate) * 100 * 0.2 +
        (1 - score_ratio) * 100 * 0.1
    )
    
    risk_labels = []
    for score in risk_score:
        if score >= 60:
            risk_labels.append('High')
        elif score >= 30:
            risk_labels.append('Medium')
        else:
            risk_labels.append('Low')
    
    # Generate target final grades
    y_grade = current_avg * 0.7 + attendance_rate * 0.2 + np.random.normal(0, 5, n_samples)
    y_grade = np.clip(y_grade, 0, 100)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    classifier.fit(X_scaled, risk_labels)
    
    # Train regressor
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    regressor.fit(X_scaled, y_grade)
    
    # Save models
    joblib.dump(classifier, f'{model_dir}/classifier.pkl')
    joblib.dump(regressor, f'{model_dir}/regressor.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    
    print("✅ Synthetic models trained and saved")
    
    return classifier, regressor, scaler

def fetch_forecasting_data(student_id=None, class_id=None):
    """
    Fetch historical data for forecasting
    Returns data for current course forecasting or future semester forecasting
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        if student_id and class_id:
            # Current course forecasting - get student's performance in specific class
            query = """
            SELECT 
                e.student_id,
                e.class_id,
                cl.course_id,
                co.course_type,
                co.course_code,
                co.course_name,
                cl.academic_year,
                cl.semester,
                -- Current performance metrics
                COALESCE(
                    (SELECT AVG((ass.score / ass2.total_points) * 100)
                     FROM assessment_scores ass
                     JOIN assessments ass2 ON ass.assessment_id = ass2.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND ass2.class_id = e.class_id
                     AND ass2.status = 'active'), 0
                ) as current_average,
                
                -- Assessment breakdown by type
                COALESCE(
                    (SELECT SUM(ass.score)
                     FROM assessment_scores ass
                     JOIN assessments a ON ass.assessment_id = a.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND a.class_id = e.class_id
                     AND a.type = 'WO'
                     AND a.status = 'active'), 0
                ) as wo_score,
                
                COALESCE(
                    (SELECT SUM(a.total_points)
                     FROM assessments a
                     WHERE a.class_id = e.class_id
                     AND a.type = 'WO'
                     AND a.status = 'active'), 0
                ) as wo_thps,
                
                COALESCE(
                    (SELECT SUM(ass.score)
                     FROM assessment_scores ass
                     JOIN assessments a ON ass.assessment_id = a.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND a.class_id = e.class_id
                     AND a.type = 'PT'
                     AND a.status = 'active'), 0
                ) as pt_score,
                
                COALESCE(
                    (SELECT SUM(a.total_points)
                     FROM assessments a
                     WHERE a.class_id = e.class_id
                     AND a.type = 'PT'
                     AND a.status = 'active'), 0
                ) as pt_thps,
                
                -- Component scores
                COALESCE(
                    (SELECT SUM(ass.score)
                     FROM assessment_scores ass
                     JOIN assessments a ON ass.assessment_id = a.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND a.class_id = e.class_id
                     AND a.type IN ('Quiz', 'Long Exam')
                     AND a.status = 'active'), 0
                ) as quizzes_score,
                
                COALESCE(
                    (SELECT SUM(a.total_points)
                     FROM assessments a
                     WHERE a.class_id = e.class_id
                     AND a.type IN ('Quiz', 'Long Exam')
                     AND a.status = 'active'), 0
                ) as quizzes_max,
                
                COALESCE(
                    (SELECT SUM(ass.score)
                     FROM assessment_scores ass
                     JOIN assessments a ON ass.assessment_id = a.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND a.class_id = e.class_id
                     AND a.type = 'Midterm Exam'
                     AND a.status = 'active'), 0
                ) as midterm_score,
                
                COALESCE(
                    (SELECT SUM(a.total_points)
                     FROM assessments a
                     WHERE a.class_id = e.class_id
                     AND a.type = 'Midterm Exam'
                     AND a.status = 'active'), 0
                ) as midterm_max,
                
                COALESCE(
                    (SELECT SUM(ass.score)
                     FROM assessment_scores ass
                     JOIN assessments a ON ass.assessment_id = a.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND a.class_id = e.class_id
                     AND a.type = 'Final Exam'
                     AND a.status = 'active'), 0
                ) as final_score,
                
                COALESCE(
                    (SELECT SUM(a.total_points)
                     FROM assessments a
                     WHERE a.class_id = e.class_id
                     AND a.type = 'Final Exam'
                     AND a.status = 'active'), 0
                ) as final_max,
                
                -- Attendance
                COALESCE(
                    (SELECT (SUM(CASE WHEN ar.status IN ('present', 'late', 'excused') THEN 1 ELSE 0 END) / 
                             NULLIF(COUNT(*), 0)) * 100
                     FROM attendance_records ar
                     WHERE ar.student_id = e.student_id 
                     AND ar.class_id = e.class_id), 100
                ) as attendance_rate,
                
                -- Assessments completed
                COALESCE(
                    (SELECT COUNT(DISTINCT ass.assessment_id)
                     FROM assessment_scores ass
                     JOIN assessments a ON ass.assessment_id = a.assessment_id
                     WHERE ass.student_id = e.student_id 
                     AND a.class_id = e.class_id
                     AND a.status = 'active'), 0
                ) as assessments_completed,
                
                -- Total assessments
                COALESCE(
                    (SELECT COUNT(*)
                     FROM assessments a
                     WHERE a.class_id = e.class_id
                     AND a.status = 'active'), 0
                ) as total_assessments,
                
                -- Grading criteria info
                gc.midterm_exam,
                gc.final_exam,
                gc.long_exams,
                gc.assignments_quizzes,
                gc.performance_tests,
                gc.projects,
                gc.recitation,
                cl.grading_mode,
                cl.has_lab_split,
                cl.lecture_weight,
                cl.lab_weight
                
            FROM enrollments e
            JOIN classes cl ON e.class_id = cl.class_id
            JOIN courses co ON cl.course_id = co.course_id
            LEFT JOIN grading_criteria gc ON cl.class_id = gc.class_id AND gc.status = 'active'
            WHERE e.student_id = ? AND e.class_id = ? AND e.status = 'enrolled'
            """
            df = pd.read_sql(query, conn, params=(student_id, class_id))
        else:
            # Future semester forecasting - get student's historical performance across all courses
            query = """
            SELECT 
                fg.student_id,
                fg.class_id,
                cl.course_id,
                co.course_type,
                co.course_code,
                cl.academic_year,
                cl.semester,
                fg.initial_grade,
                fg.final_grade,
                fg.remarks,
                -- Course difficulty indicators
                (SELECT COUNT(DISTINCT e2.student_id)
                 FROM enrollments e2
                 WHERE e2.class_id = cl.class_id) as class_size,
                -- Student's year level at time of course
                s.year_level as student_year_level
            FROM final_grades fg
            JOIN classes cl ON fg.class_id = cl.class_id
            JOIN courses co ON cl.course_id = co.course_id
            JOIN enrollments e ON fg.student_id = e.student_id AND fg.class_id = e.class_id
            JOIN students s ON fg.student_id = s.student_id
            WHERE fg.student_id = ?
            ORDER BY cl.academic_year DESC, cl.semester DESC, co.course_code
            LIMIT 50
            """
            df = pd.read_sql(query, conn, params=(student_id,))
        
        conn.close()
        return df if not df.empty else None
        
    except Error as e:
        print(f"Error fetching forecasting data: {e}")
        if conn:
            conn.close()
        return None

if __name__ == '__main__':
    train_models_with_real_data()

