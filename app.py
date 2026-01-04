"""
Machine Learning Service for Academic Performance Prediction
Uses Gradient Boosting for risk classification and grade prediction
Enhanced with versioning, comprehensive metrics, and training logs
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from train_enhanced import train_models_enhanced, get_db_connection

app = Flask(__name__)
CORS(app)

# Initialize models
classifier = None
regressor = None
scaler = None
model_version = "1.0"
active_model_id = None

def initialize_models():
    """Initialize or load pre-trained models"""
    global classifier, regressor, scaler, model_version, active_model_id
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Try to load active model from database
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_id, model_version, file_path 
                FROM ml_models 
                WHERE is_active = 1 AND model_type = 'both'
                ORDER BY activated_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                active_model_id, model_version, file_path = result
                # Try to load models with version
                classifier_path = file_path.replace('regressor', 'classifier')
                scaler_path = file_path.replace('regressor', 'scaler')
                
                if os.path.exists(classifier_path) and os.path.exists(file_path) and os.path.exists(scaler_path):
                    classifier = joblib.load(classifier_path)
                    regressor = joblib.load(file_path)
                    scaler = joblib.load(scaler_path)
                    print(f"Models v{model_version} loaded successfully")
                    cursor.close()
                    conn.close()
                    return
            cursor.close()
        except Exception as e:
            print(f"Error loading model from database: {e}")
        finally:
            conn.close()
    
    # Fallback: Try to load existing models (legacy)
    if os.path.exists(f'{model_dir}/classifier.pkl') and os.path.exists(f'{model_dir}/regressor.pkl'):
        classifier = joblib.load(f'{model_dir}/classifier.pkl')
        regressor = joblib.load(f'{model_dir}/regressor.pkl')
        scaler = joblib.load(f'{model_dir}/scaler.pkl')
        print("Legacy models loaded successfully")
    else:
        # Train new models with synthetic data
        print("Training new models...")
        train_models()
        print("Models trained successfully")

def train_models(use_real_data=True):
    """Train models with real database data or synthetic data"""
    global classifier, regressor, scaler
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Try to use real data first
    if use_real_data:
        try:
            from train_model import train_models_with_real_data
            classifier, regressor, scaler = train_models_with_real_data()
            return
        except Exception as e:
            print(f"Failed to train with real data: {e}")
            print("Falling back to synthetic data...")
    
    # Fallback to synthetic data
    train_models_synthetic()

def train_models_synthetic():
    """Train models with synthetic data based on academic patterns"""
    global classifier, regressor, scaler
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: current_average, attendance_rate, assessments_completed, total_score, total_max
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
    
    # Generate risk labels (High, Medium, Low)
    # Risk increases with lower average, lower attendance, fewer assessments
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
    
    # Generate target final grades (regression)
    # Based on current performance with some variance
    y_grade = current_avg * 0.7 + attendance_rate * 0.2 + np.random.normal(0, 5, n_samples)
    y_grade = np.clip(y_grade, 0, 100)
    
    # Calculate required score to pass (60% minimum)
    required_scores = []
    for i in range(n_samples):
        remaining_weight = 0.3  # Assume 30% of grade remaining
        current_weighted = current_avg[i] * 0.7
        needed = (60 - current_weighted) / remaining_weight
        required_scores.append(max(0, min(100, needed)))
    
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
    
    # Train regressor for grade prediction
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
    
    print("Synthetic models trained and saved successfully")

def predict_risk(data):
    """Predict student risk level and required scores with probability-based predictions"""
    global classifier, regressor, scaler, model_version, active_model_id
    
    # Extract features
    current_avg = data.get('current_average', 75)
    attendance_rate = data.get('attendance_rate', 90)
    assessments_completed = data.get('assessments_completed', 10)
    total_score = data.get('total_score', 0)
    total_max = data.get('total_max', 100)
    
    # Calculate derived features
    completion_rate = min(1.0, assessments_completed / 30.0)
    score_ratio = total_score / total_max if total_max > 0 else 0
    
    # Prepare feature vector
    X = np.array([[
        current_avg,
        attendance_rate,
        completion_rate,
        score_ratio,
        assessments_completed
    ]])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict risk level with probabilities
    risk_level = classifier.predict(X_scaled)[0]
    risk_proba = classifier.predict_proba(X_scaled)[0]
    
    # Get probability distribution for all risk levels
    risk_classes = classifier.classes_
    risk_probabilities = {}
    for i, risk_class in enumerate(risk_classes):
        risk_probabilities[risk_class.lower()] = round(float(risk_proba[i] * 100), 2)
    
    # Get risk score (probability of high risk)
    risk_score = 0
    if 'High' in risk_classes:
        high_idx = list(risk_classes).index('High')
        risk_score = risk_proba[high_idx] * 100
    
    # Predict final grade with confidence intervals
    predicted_grade = regressor.predict(X_scaled)[0]
    predicted_grade = max(0, min(100, predicted_grade))
    
    # Get prediction probabilities for grade ranges
    # Estimate confidence based on model's prediction variance
    grade_probabilities = {
        'passing_60_70': 0.0,
        'passing_70_80': 0.0,
        'passing_80_90': 0.0,
        'passing_90_100': 0.0,
        'failing_below_60': 0.0
    }
    
    if predicted_grade >= 90:
        grade_probabilities['passing_90_100'] = 100.0
    elif predicted_grade >= 80:
        grade_probabilities['passing_80_90'] = 100.0
    elif predicted_grade >= 70:
        grade_probabilities['passing_70_80'] = 100.0
    elif predicted_grade >= 60:
        grade_probabilities['passing_60_70'] = 100.0
    else:
        grade_probabilities['failing_below_60'] = 100.0
    
    # Calculate required score to pass (assuming 60% is passing)
    remaining_weight = 0.3
    current_weighted = current_avg * 0.7
    required_score = (60 - current_weighted) / remaining_weight if remaining_weight > 0 else 60
    required_score = max(0, min(100, required_score))
    
    # Calculate minimum grade percentage required to pass
    min_grade_to_pass = max(0, min(100, required_score))
    
    return {
        'risk_level': risk_level,
        'risk_score': round(float(risk_score), 2),
        'risk_probabilities': risk_probabilities,
        'predicted_final_grade': round(float(predicted_grade), 2),
        'grade_probabilities': grade_probabilities,
        'required_score_to_pass': round(float(required_score), 2),
        'minimum_grade_to_pass_percentage': round(float(min_grade_to_pass), 2),
        'model_version': model_version,
        'model_id': active_model_id
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction = predict_risk(data)
        return jsonify(prediction), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_version': model_version,
        'models_loaded': classifier is not None and regressor is not None
    }), 200

@app.route('/train', methods=['POST'])
def train():
    """Enhanced training endpoint with versioning and comprehensive metrics"""
    try:
        data = request.get_json() or {}
        use_real_data = data.get('use_real_data', True)
        triggered_by_user_id = data.get('user_id', None)
        uploaded_data = data.get('dataset', None)
        
        # Handle uploaded dataset
        df = None
        data_source = 'real'
        if uploaded_data:
            # Parse uploaded CSV/JSON data
            if isinstance(uploaded_data, str):
                # CSV string
                from io import StringIO
                df = pd.read_csv(StringIO(uploaded_data))
            elif isinstance(uploaded_data, list):
                # JSON array
                df = pd.DataFrame(uploaded_data)
            elif isinstance(uploaded_data, dict):
                # JSON object
                df = pd.DataFrame([uploaded_data])
            data_source = 'uploaded'
        elif not use_real_data:
            data_source = 'synthetic'
        
        # Train models using enhanced training
        result = train_models_enhanced(
            df=df,
            data_source=data_source,
            triggered_by_user_id=triggered_by_user_id,
            use_uploaded_data=uploaded_data is not None
        )
        
        if result.get('success'):
            # Reload models after training
            initialize_models()
            return jsonify({
                'success': True,
                'message': 'Models trained successfully',
                'model_version': result.get('model_version'),
                'training_log_id': result.get('training_log_id'),
                'metrics': {
                    'classifier': result.get('classifier', {}).get('metrics', {}),
                    'regressor': result.get('regressor', {}).get('metrics', {})
                },
                'dataset_size': result.get('dataset_size'),
                'training_duration_seconds': result.get('training_duration_seconds')
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Training failed'),
                'errors': result.get('errors', []),
                'training_log_id': result.get('training_log_id')
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain models endpoint (alias for train)"""
    return train()

@app.route('/training/status/<int:training_log_id>', methods=['GET'])
def get_training_status(training_log_id):
    """Get training status and metrics for a specific training log"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                training_log_id,
                training_status,
                data_source,
                dataset_size,
                training_samples,
                validation_samples,
                classifier_accuracy,
                classifier_mae,
                classifier_rmse,
                classifier_confusion_matrix,
                classifier_precision,
                classifier_recall,
                classifier_f1_score,
                regressor_mae,
                regressor_rmse,
                regressor_r2_score,
                regressor_mean_error,
                model_version,
                previous_model_version,
                training_duration_seconds,
                error_message,
                training_notes,
                triggered_by_user_id,
                created_at,
                completed_at
            FROM ml_training_logs
            WHERE training_log_id = %s
        """, (training_log_id,))
        
        log = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not log:
            return jsonify({'error': 'Training log not found'}), 404
        
        # Parse confusion matrix if present
        if log.get('classifier_confusion_matrix'):
            try:
                log['classifier_confusion_matrix'] = json.loads(log['classifier_confusion_matrix'])
            except:
                pass
        
        return jsonify(log), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training/logs', methods=['GET'])
def get_training_logs():
    """Get list of training logs with pagination"""
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                training_log_id,
                training_status,
                data_source,
                dataset_size,
                model_version,
                training_duration_seconds,
                created_at,
                completed_at
            FROM ml_training_logs
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        
        logs = cursor.fetchall()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM ml_training_logs")
        total = cursor.fetchone()['total']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'logs': logs,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/active', methods=['GET'])
def get_active_model():
    """Get information about the currently active model"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                m.model_id,
                m.model_version,
                m.model_type,
                m.file_path,
                m.is_active,
                m.created_at,
                m.activated_at,
                tl.training_status,
                tl.classifier_accuracy,
                tl.regressor_rmse,
                tl.dataset_size
            FROM ml_models m
            LEFT JOIN ml_training_logs tl ON m.training_log_id = tl.training_log_id
            WHERE m.is_active = 1
            ORDER BY m.activated_at DESC
            LIMIT 1
        """)
        
        model = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not model:
            return jsonify({
                'active': False,
                'message': 'No active model found'
            }), 200
        
        return jsonify({
            'active': True,
            'model': model
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/versions', methods=['GET'])
def get_model_versions():
    """Get list of all model versions"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                m.model_id,
                m.model_version,
                m.model_type,
                m.is_active,
                m.created_at,
                m.activated_at,
                tl.training_status,
                tl.classifier_accuracy,
                tl.regressor_rmse
            FROM ml_models m
            LEFT JOIN ml_training_logs tl ON m.training_log_id = tl.training_log_id
            ORDER BY m.created_at DESC
        """)
        
        models = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({'models': models}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/<int:model_id>/activate', methods=['POST'])
def activate_model(model_id):
    """Activate a specific model version"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Get model info
        cursor.execute("SELECT model_version, file_path, model_type FROM ml_models WHERE model_id = %s", (model_id,))
        model = cursor.fetchone()
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        model_version, file_path, model_type = model
        
        # Deactivate all models of same type
        cursor.execute("UPDATE ml_models SET is_active = 0 WHERE model_type IN ('both', %s)", (model_type,))
        
        # Activate selected model
        cursor.execute("""
            UPDATE ml_models 
            SET is_active = 1, activated_at = NOW() 
            WHERE model_id = %s
        """, (model_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Reload models
        initialize_models()
        
        return jsonify({
            'success': True,
            'message': f'Model v{model_version} activated successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def forecast_current_course(student_id, class_id):
    """
    Forecast final grade for a course in progress
    Uses current assessment scores and historical patterns
    """
    try:
        from train_model import fetch_forecasting_data, get_db_connection
        import json
        
        # Fetch current course data
        df = fetch_forecasting_data(student_id=student_id, class_id=class_id)
        if df is None or df.empty:
            return {'error': 'No data available for forecasting'}
        
        row = df.iloc[0]
        
        # Get class and grading info
        conn = get_db_connection()
        if not conn:
            return {'error': 'Database connection failed'}
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT c.*, co.course_type, co.course_code, co.course_name,
                   gc.*, cl.grading_mode, cl.has_lab_split, cl.lecture_weight, cl.lab_weight
            FROM classes c
            JOIN courses co ON c.course_id = co.course_id
            LEFT JOIN grading_criteria gc ON c.class_id = gc.class_id AND gc.status = 'active'
            LEFT JOIN classes cl ON c.class_id = cl.class_id
            WHERE c.class_id = %s
        """, (class_id,))
        class_info = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not class_info:
            return {'error': 'Class not found'}
        
        # Calculate forecast based on grading system
        grading_mode = class_info.get('grading_mode', 'component_based')
        course_type = class_info.get('course_type', '')
        
        # Use existing regressor for base prediction
        current_avg = float(row.get('current_average', 0))
        attendance_rate = float(row.get('attendance_rate', 100))
        assessments_completed = int(row.get('assessments_completed', 0))
        total_assessments = int(row.get('total_assessments', 1))
        
        # Calculate completion percentage
        completion_rate = assessments_completed / total_assessments if total_assessments > 0 else 0
        
        # Prepare features for ML prediction
        score_ratio = current_avg / 100.0 if current_avg > 0 else 0
        
        X = np.array([[
            current_avg,
            attendance_rate,
            min(1.0, completion_rate),
            score_ratio,
            assessments_completed
        ]])
        
        X_scaled = scaler.transform(X)
        predicted_equivalent_grade = regressor.predict(X_scaled)[0]
        predicted_equivalent_grade = max(0, min(100, predicted_equivalent_grade))
        
        # Calculate initial grade based on grading system
        if grading_mode == 'wo_pt':
            # WO/PT system
            wo_score = float(row.get('wo_score', 0))
            wo_thps = float(row.get('wo_thps', 1))
            pt_score = float(row.get('pt_score', 0))
            pt_thps = float(row.get('pt_thps', 1))
            
            # Determine weights based on course type
            if course_type in ['Language', 'GE', 'Social Science']:
                wo_weight, pt_weight = 40, 60
            elif course_type in ['Business', 'Engineering', 'Natural Science']:
                wo_weight, pt_weight = 50, 50
            elif course_type == 'Laboratory':
                wo_weight, pt_weight = 30, 70
            else:
                wo_weight, pt_weight = 50, 50
            
            wo_ratio = (wo_score / wo_thps) if wo_thps > 0 else 0
            pt_ratio = (pt_score / pt_thps) if pt_thps > 0 else 0
            predicted_initial_grade = (wo_ratio * wo_weight) + (pt_ratio * pt_weight)
            
            # Calculate equivalent grade
            if course_type == 'Accountancy':
                predicted_equivalent_grade = (predicted_initial_grade * 0.65) + 35
            else:
                predicted_equivalent_grade = (predicted_initial_grade * 0.60) + 40
        else:
            # Component-based system - use weighted calculation
            quizzes_score = float(row.get('quizzes_score', 0))
            quizzes_max = float(row.get('quizzes_max', 1))
            midterm_score = float(row.get('midterm_score', 0))
            midterm_max = float(row.get('midterm_max', 1))
            final_score = float(row.get('final_score', 0))
            final_max = float(row.get('final_max', 1))
            
            # Get weights from grading criteria or defaults
            midterm_weight = float(class_info.get('midterm_exam', 25))
            final_weight = float(class_info.get('final_exam', 35))
            quizzes_weight = float(class_info.get('assignments_quizzes', 7.5) or class_info.get('long_exams', 20))
            
            # Calculate component contributions
            quizzes_contrib = (quizzes_score / quizzes_max * quizzes_weight) if quizzes_max > 0 else 0
            midterm_contrib = (midterm_score / midterm_max * midterm_weight) if midterm_max > 0 else 0
            final_contrib = (final_score / final_max * final_weight) if final_max > 0 else 0
            
            # Estimate remaining components (if not yet taken)
            remaining_weight = 100 - (quizzes_weight + midterm_weight + final_weight)
            if remaining_weight > 0:
                # Estimate based on current average
                estimated_remaining = (current_avg / 100) * remaining_weight
            else:
                estimated_remaining = 0
            
            predicted_initial_grade = quizzes_contrib + midterm_contrib + final_contrib + estimated_remaining
            predicted_equivalent_grade = predicted_initial_grade  # Component-based uses IG directly
        
        # Transmute grade
        from train_model import get_db_connection
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT transmutation_table FROM grading_systems 
                WHERE is_active = 1 AND status = 'active' LIMIT 1
            """)
            result = cursor.fetchone()
            transmutation_table = result['transmutation_table'] if result else None
            cursor.close()
            conn.close()
        else:
            transmutation_table = None
        
        # Simple transmutation (can be enhanced)
        if predicted_equivalent_grade >= 97:
            transmuted = '1.0'
        elif predicted_equivalent_grade >= 93:
            transmuted = '1.25'
        elif predicted_equivalent_grade >= 89:
            transmuted = '1.50'
        elif predicted_equivalent_grade >= 85:
            transmuted = '1.75'
        elif predicted_equivalent_grade >= 80:
            transmuted = '2.0'
        elif predicted_equivalent_grade >= 75:
            transmuted = '2.25'
        elif predicted_equivalent_grade >= 70:
            transmuted = '2.5'
        elif predicted_equivalent_grade >= 65:
            transmuted = '2.75'
        elif predicted_equivalent_grade >= 60:
            transmuted = '3.0'
        elif predicted_equivalent_grade >= 55:
            transmuted = '4.0'
        else:
            transmuted = '5.0'
        
        # Calculate confidence based on data completeness
        data_completeness = (assessments_completed / total_assessments) if total_assessments > 0 else 0
        confidence_score = min(100, max(50, data_completeness * 100))
        
        # Confidence interval (simple estimation)
        std_dev = 5.0  # Estimated standard deviation
        confidence_interval_lower = max(0, predicted_equivalent_grade - (1.96 * std_dev))
        confidence_interval_upper = min(100, predicted_equivalent_grade + (1.96 * std_dev))
        
        return {
            'success': True,
            'predicted_initial_grade': round(float(predicted_initial_grade), 2),
            'predicted_equivalent_grade': round(float(predicted_equivalent_grade), 2),
            'predicted_transmuted_grade': transmuted,
            'confidence_score': round(float(confidence_score), 2),
            'confidence_interval_lower': round(float(confidence_interval_lower), 2),
            'confidence_interval_upper': round(float(confidence_interval_upper), 2),
            'forecast_method': 'gradient_boosting_with_formula',
            'model_version': model_version,
            'data_completeness': round(float(data_completeness * 100), 2),
            'component_breakdown': {
                'current_average': round(float(current_avg), 2),
                'attendance_rate': round(float(attendance_rate), 2),
                'assessments_completed': int(assessments_completed),
                'total_assessments': int(total_assessments)
            }
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/forecast/current-course', methods=['POST'])
def forecast_current_course_endpoint():
    """Forecast final grade for current course"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        class_id = data.get('class_id')
        
        if not student_id or not class_id:
            return jsonify({'error': 'student_id and class_id are required'}), 400
        
        result = forecast_current_course(int(student_id), int(class_id))
        return jsonify(result), 200 if result.get('success') else 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecast/future-semester', methods=['POST'])
def forecast_future_semester():
    """Predict grades for upcoming courses based on historical performance"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        course_ids = data.get('course_ids', [])  # List of upcoming course IDs
        
        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400
        
        from train_model import fetch_forecasting_data, get_db_connection
        
        # Fetch historical performance
        df = fetch_forecasting_data(student_id=student_id)
        if df is None or df.empty:
            return jsonify({'error': 'No historical data available'}), 400
        
        # Calculate average performance by course type (use IG% as primary numeric signal)
        if 'equivalent_grade' in df.columns:
            avg_by_type = df.groupby('course_type')['equivalent_grade'].mean().to_dict()
        else:
            avg_by_type = df.groupby('course_type')['initial_grade'].mean().to_dict()
        
        # Predict for each upcoming course
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        predictions = []
        if course_ids:
            cursor = conn.cursor(dictionary=True)
            for course_id in course_ids:
                cursor.execute("""
                    SELECT course_id, course_code, course_name, course_type
                    FROM courses WHERE course_id = %s
                """, (course_id,))
                course = cursor.fetchone()
                if course:
                    course_type = course['course_type']
                    if 'equivalent_grade' in df.columns:
                        avg_grade = avg_by_type.get(course_type, df['equivalent_grade'].mean())
                    else:
                        avg_grade = avg_by_type.get(course_type, df['initial_grade'].mean())
                    
                    # Add some variance based on historical performance
                    if 'equivalent_grade' in df.columns:
                        std_dev = df['equivalent_grade'].std() if len(df) > 1 else 10.0
                    else:
                        std_dev = df['initial_grade'].std() if len(df) > 1 else 10.0
                    predicted = max(0, min(100, avg_grade + np.random.normal(0, std_dev * 0.3)))
                    
                    predictions.append({
                        'course_id': course_id,
                        'course_code': course['course_code'],
                        'course_name': course['course_name'],
                        'predicted_equivalent_grade': round(float(predicted), 2),
                        'confidence_score': 70.0  # Lower confidence for future predictions
                    })
            cursor.close()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'historical_average': round(float((df['equivalent_grade'] if 'equivalent_grade' in df.columns else df['initial_grade']).mean()), 2),
            'model_version': model_version
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=True)

