# ML Service for Academic Performance Prediction

This Python Flask service provides machine learning predictions for student academic performance.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python app.py
```

The service will run on `http://localhost:5000`

## Endpoints

- `POST /predict` - Get risk prediction for a student
- `GET /health` - Health check
- `POST /retrain` - Retrain models

## Prediction Request Format

```json
{
    "current_average": 75.5,
    "attendance_rate": 90.0,
    "assessments_completed": 12,
    "total_score": 450,
    "total_max": 600
}
```

## Response Format

```json
{
    "risk_level": "Medium",
    "risk_score": 35.5,
    "predicted_final_grade": 78.2,
    "required_score_to_pass": 55.0,
    "model_version": "1.0"
}
```

