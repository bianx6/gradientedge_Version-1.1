# ML Model Training Guide

This guide explains how to train the machine learning models using real database data.

## Overview

The ML service uses two models:
1. **Risk Classifier**: Predicts student risk level (High, Medium, Low)
2. **Grade Regressor**: Predicts final grade percentage

## Training Methods

### Method 1: Train with Real Database Data (Recommended)

The system will automatically fetch real student performance data from your database and train the models.

**Features:**
- Uses actual student grades, attendance, and assessment data
- More accurate predictions based on real patterns
- Automatically falls back to synthetic data if insufficient real data is available

**Requirements:**
- Database must have student performance data (grades, attendance, assessments)
- At least 10-20 complete records for meaningful training
- MySQL connector installed: `pip install mysql-connector-python`

### Method 2: Train with Synthetic Data (Fallback)

If real data is not available, the system uses synthetic data based on academic patterns.

**Features:**
- Works immediately without database data
- Good for initial setup and testing
- Less accurate than real data training

## Training Options

### Option 1: Via Python Script (Direct)

Run the training script directly:

```bash
cd ml_service
python train_model.py
```

This will:
1. Connect to your database (using config from `config/database.php`)
2. Fetch real student performance data
3. Train both models
4. Save models to `ml_service/models/` directory
5. Display training metrics (accuracy, RMSE)

### Option 2: Via ML Service API

Start the ML service and call the retrain endpoint:

```bash
# Terminal 1: Start ML service
cd ml_service
python app.py

# Terminal 2: Call retrain endpoint
curl -X POST http://localhost:5000/retrain \
  -H "Content-Type: application/json" \
  -d '{"use_real_data": true}'
```

### Option 3: Via PHP Admin Interface

1. Log in as admin
2. Navigate to ML Training page (if available)
3. Click "Train Models" button
4. Select "Use Real Data" or "Use Synthetic Data"
5. Wait for training to complete

### Option 4: Via API Endpoint

```bash
curl -X POST https://gradientedge1.gradientedge.online/api/train_ml_model.php \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "use_real_data=1"
```

## Database Configuration

The training script automatically reads database configuration from `config/database.php`:

```php
define('DB_HOST', 'localhost');
define('DB_USER', 'your_user');
define('DB_PASS', 'your_password');
define('DB_NAME', 'your_database');
```

If the script cannot parse the PHP config, it will use defaults:
- Host: localhost
- User: root
- Password: (empty)
- Database: academic_performance
- Port: 3306

## Training Data Requirements

### Minimum Data for Real Training

**For Risk Classification:**
- At least 10-20 students with complete final grades
- Students with different risk levels (High, Medium, Low)
- Mix of performance levels

**For Grade Prediction:**
- At least 10-20 students with final grades
- Students with various current averages
- Attendance records
- Assessment scores

### Data Quality

Better training results with:
- More data (100+ students ideal)
- Balanced risk levels (not all High or all Low)
- Complete records (grades, attendance, assessments)
- Recent data (last 1-2 academic years)

## Training Process

1. **Data Fetching**: Queries database for student performance data
2. **Feature Engineering**: Calculates derived features (completion rate, score ratio)
3. **Data Splitting**: Splits into training (80%) and testing (20%) sets
4. **Model Training**: Trains Gradient Boosting models
5. **Evaluation**: Calculates accuracy (classification) and RMSE (regression)
6. **Model Saving**: Saves trained models to `models/` directory

## Model Files

After training, models are saved to:
- `ml_service/models/classifier.pkl` - Risk classification model
- `ml_service/models/regressor.pkl` - Grade prediction model
- `ml_service/models/scaler.pkl` - Feature scaler

## Training Output

Example training output:

```
============================================================
Training ML Models with Real Database Data
============================================================

Fetched 245 training samples from database

📊 Training Risk Classification Model...
   Training samples: 245
   ✅ Risk Classification Accuracy: 87.76%
   📈 Classification Report:
              precision    recall  f1-score   support
        High       0.89      0.85      0.87        34
         Low       0.92      0.95      0.93       145
      Medium       0.82      0.79      0.80        66
   💾 Classifier saved to models/classifier.pkl

📊 Training Grade Prediction Model...
   Training samples: 245
   ✅ Grade Prediction RMSE: 8.45
   📊 Mean Error: 6.23 points
   💾 Regressor saved to models/regressor.pkl
   💾 Scaler saved to models/scaler.pkl

============================================================
✅ Model Training Complete!
============================================================
```

## Troubleshooting

### "No training data found"
- Check database connection settings
- Verify students have grades and attendance records
- Ensure `final_grades` table has data

### "Insufficient data for classification/regression"
- Need at least 10-20 complete records
- System will use synthetic data as fallback
- Add more student data to database

### "MySQL connector not available"
```bash
pip install mysql-connector-python
```

### "Error connecting to MySQL"
- Verify database credentials in `config/database.php`
- Check database server is running
- Test connection manually

### Models not improving
- Add more training data
- Ensure data quality (no missing values)
- Check for data balance (not all same risk level)

## Retraining Schedule

**Recommended:**
- **Initial Training**: After collecting 50+ student records
- **Regular Retraining**: Monthly or quarterly
- **After Major Changes**: When grading system changes

**Automatic Retraining:**
- Can be scheduled via cron job
- Call `/api/train_ml_model.php` periodically
- Or use ML service `/retrain` endpoint

## Best Practices

1. **Train with Real Data**: Always prefer real data over synthetic
2. **Regular Updates**: Retrain monthly to capture new patterns
3. **Monitor Performance**: Check accuracy metrics after training
4. **Data Quality**: Ensure clean, complete data before training
5. **Version Control**: Keep track of model versions and training dates
6. **Backup Models**: Save model files before retraining

## Next Steps

After training:
1. Test predictions with sample students
2. Monitor prediction accuracy
3. Adjust model parameters if needed
4. Schedule regular retraining

