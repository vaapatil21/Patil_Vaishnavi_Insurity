# TelematicsAI - Insurance Risk Assessment & Policy Integration

An AI-powered telematics solution that analyzes driver behavior and automatically integrates with insurance policy systems for real-time premium adjustments.

## Overview

TelematicsAI transforms traditional insurance pricing by using machine learning to assess driver risk based on actual driving behavior, then automatically updating policy premiums in real-time. The system processes telematics data (speed, braking, acceleration) through an XGBoost model and delivers immediate customer value through automated policy integration.

**Key Features:**
- Real-time driving behavior analysis using XGBoost ML model
- Automated policy system integration via RESTful APIs
- Interactive dashboard for demonstrations and monitoring
- Batch processing capabilities for enterprise scale
- Comprehensive health monitoring and analytics

## Architecture

```
Telematics Data → AI Risk Assessment → Policy Integration → Customer Savings
      ↓                    ↓                    ↓               ↓
GPS/Accelerometer    XGBoost Model      RESTful APIs    Real-time Updates
```

<details>
<summary><strong>Prerequisites</strong></summary>

### System Requirements
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- Port 8000 and 8501 available

### Required Python Packages
```bash
fastapi>=0.68.0
uvicorn>=0.15.0
streamlit>=1.28.0
scikit-learn>=1.0.0
xgboost>=1.5.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.0.0
requests>=2.25.0
python-dotenv>=0.19.0
httpx>=0.24.0
python-multipart>=0.0.6
```

</details>

<details>
<summary><strong>Quick Start</strong></summary>

### 1. Clone the Repository
```bash
git clone https://github.com/vaapatil21/Patil_Vaishnavi_Insurity.git
cd Patil_Vaishnavi_Insurity
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment
Create a `.env` file in the project root:
```bash
# API Configuration
API_KEY=telematics_ai_secret_key_2024
POLICY_SYSTEM_URL=https://policy-system.company.com/api
POLICY_SYSTEM_API_KEY=demo_policy_system_key
LOG_LEVEL=INFO
MODEL_PATH=models/discount_eligibility_model.pkl
```

### 4. Start the API Server
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Launch the Dashboard
Open a new terminal and run:
```bash
streamlit run src/dashboard.py
```

### 6. Access the Application
- **API Documentation**: http://localhost:8000/docs
- **Interactive Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

</details>

<details>
<summary><strong> Project Structure</strong></summary>

```
telematics-ai/
├── .env                          # Environment variables
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── src/
│   ├── api.py                   # FastAPI application with ML endpoints
│   ├── dashboard.py             # Streamlit dashboard interface
|   ├── data_simulator.py
|   ├── feature_engineering.py
|   ├── model_comparison.py
|   ├── quick_accuracy_check.py
│   └── train_model.py
├── models/
│   └── discount_eligibility_model.pkl  # Trained XGBoost model...etc
├── data/
│   ├──telematics_data.csv   # Sample telematics data
|   ├── business_analysis.csv
|   ├── ml_ready_dataset.csv
|   └── model_comparison_results.csv
└── docs/
    └── ModelDevelopment.ipynb           # Integration documentation
└── bin/
    └── run.sh
```

</details>

<details>
<summary><strong>API Usage</strong></summary>

### Authentication
All API endpoints require authentication via API key in the request header:
```bash
X-API-KEY: telematics_ai_secret_key_2024
```

### Core Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Risk Prediction
```bash
POST /predict
Content-Type: application/json

{
  "avg_speed_kph": 65.0,
  "max_speed_kph": 85.0,
  "harsh_braking_events": 2,
  "harsh_acceleration_events": 1,
  "speeding_events": 3,
  "phone_usage_events": 0,
  "total_distance_km": 1200.0,
  "total_trips": 45,
  "night_driving_ratio": 0.15,
  "weekend_driving_ratio": 0.25,
  "avg_trip_duration": 25.0,
  "harsh_cornering_events": 1
}
```

#### 3. Policy Integration
```bash
POST /integrate/predict-and-update
Content-Type: application/json

{
  "driver_id": "DRV001",
  "policy_id": "POL001",
  "driver_features": { ... },
  "update_policy": true
}
```

#### 4. Batch Processing
```bash
POST /integrate/batch
Content-Type: application/json

{
  "drivers": [
    {
      "driver_id": "DRV001",
      "policy_id": "POL001", 
      "driver_features": { ... },
      "update_policy": true
    }
  ]
}
```

### Example cURL Commands

```bash
# Health check
curl -X GET "http://localhost:8000/health" \
  -H "X-API-KEY: telematics_ai_secret_key_2024"

# Risk prediction
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-KEY: telematics_ai_secret_key_2024" \
  -H "Content-Type: application/json" \
  -d '{
    "avg_speed_kph": 65.0,
    "max_speed_kph": 85.0,
    "harsh_braking_events": 2,
    "harsh_acceleration_events": 1,
    "speeding_events": 3,
    "phone_usage_events": 0,
    "total_distance_km": 1200.0,
    "total_trips": 45,
    "night_driving_ratio": 0.15,
    "weekend_driving_ratio": 0.25,
    "avg_trip_duration": 25.0,
    "harsh_cornering_events": 1
  }'
```

</details>

<details>
<summary><strong>💻 Dashboard Features</strong></summary>

### Navigation Sections

#### 1. Overview
- System architecture visualization
- Business impact metrics
- Recent activity monitoring

#### 2. AI Risk Prediction  
- Interactive driver behavior input
- Real-time risk assessment
- Preset driver profiles (Safe, Average, Risky)
- Risk factor breakdown analysis

#### 3. Policy Integration
- Live demonstration of AI-to-policy integration
- Real-time premium calculations
- Customer savings visualization
- Integration status monitoring

#### 4. Batch Processing
- Enterprise-scale processing capabilities
- Multiple driver processing simulation
- Performance metrics and throughput analysis

#### 5. Analytics Dashboard
- Model performance metrics
- Business intelligence insights
- Feature importance analysis
- ROI calculations

#### 6. System Monitoring
- Health checks and status monitoring
- Performance metrics
- Error logs and system events
- Configuration management

### Demo Features
- Preset driver profiles for quick demonstrations
- Real-time API integration
- Professional visualizations with Plotly
- Comprehensive system status monitoring

</details>

<details>
<summary><strong>🔬 Model Information</strong></summary>

### XGBoost Risk Assessment Model

**Model Type**: XGBoost Classifier  
**Accuracy**: 83.2% on validation set  
**Features**: 18 behavioral and derived features  
**Output**: Binary classification (discount eligible/not eligible)

#### Feature Engineering
The model uses 18 engineered features derived from raw telematics data:

**Primary Features:**
- Average and maximum speed
- Harsh braking/acceleration events
- Speeding violations
- Phone usage while driving
- Night and weekend driving ratios
- Trip patterns and duration

**Derived Features:**
- Speed safety scores
- Braking safety scores
- Acceleration safety scores
- Consistency scores
- Overall safety composite score

#### Model Performance
- Precision: 82%
- Recall: 78% 
- F1-Score: 80%
- Response Time: <200ms

#### Discount Tiers
- **Gold Tier (30% off)**: Confidence ≥ 80%
- **Silver Tier (20% off)**: Confidence ≥ 60%
- **Bronze Tier (10% off)**: Confidence ≥ 50%
- **Standard Rate**: Confidence < 50%

</details>

<details>
<summary><strong>⚙️ Configuration</strong></summary>

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API authentication key | `telematics_ai_secret_key_2024` |
| `POLICY_SYSTEM_URL` | External policy system endpoint | `https://policy-system.company.com/api` |
| `POLICY_SYSTEM_API_KEY` | Policy system authentication | `demo_policy_system_key` |
| `LOG_LEVEL` | Logging level (INFO, DEBUG, ERROR) | `INFO` |
| `MODEL_PATH` | Path to trained model file | `models/discount_eligibility_model.pkl` |

### API Configuration
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **Timeout**: 30 seconds
- **Max Request Size**: 10MB
- **Rate Limiting**: 100 requests/hour per API key

### Dashboard Configuration  
- **Host**: localhost
- **Port**: 8501
- **Auto-refresh**: Configurable per page
- **Themes**: Light/Dark mode support

</details>

<details>
<summary><strong>🧪 Testing</strong></summary>

### API Testing

#### Using Python
```python
import requests

# Test health endpoint
response = requests.get(
    "http://localhost:8000/health",
    headers={"X-API-KEY": "telematics_ai_secret_key_2024"}
)
print(response.json())

# Test prediction endpoint
test_data = {
    "avg_speed_kph": 65.0,
    "max_speed_kph": 85.0,
    "harsh_braking_events": 2,
    "harsh_acceleration_events": 1,
    "speeding_events": 3,
    "phone_usage_events": 0,
    "total_distance_km": 1200.0,
    "total_trips": 45,
    "night_driving_ratio": 0.15,
    "weekend_driving_ratio": 0.25,
    "avg_trip_duration": 25.0,
    "harsh_cornering_events": 1
}

response = requests.post(
    "http://localhost:8000/predict",
    json=test_data,
    headers={"X-API-KEY": "telematics_ai_secret_key_2024"}
)
print(response.json())
```

#### Using curl
```bash
# Test integration endpoint
curl -X POST "http://localhost:8000/integrate/predict-and-update" \
  -H "X-API-KEY: telematics_ai_secret_key_2024" \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": "TEST_DRIVER",
    "policy_id": "TEST_POLICY",
    "driver_features": {
      "avg_speed_kph": 65.0,
      "max_speed_kph": 85.0,
      "harsh_braking_events": 2,
      "harsh_acceleration_events": 1,
      "speeding_events": 3,
      "phone_usage_events": 0,
      "total_distance_km": 1200.0,
      "total_trips": 45,
      "night_driving_ratio": 0.15,
      "weekend_driving_ratio": 0.25,
      "avg_trip_duration": 25.0,
      "harsh_cornering_events": 1
    },
    "update_policy": true
  }'
```

### Dashboard Testing
1. Navigate to http://localhost:8501
2. Use the sidebar navigation to test each section
3. Try different driver profiles in the Policy Integration section
4. Monitor system status in the Monitoring section

</details>

<details>
<summary><strong>🐛 Troubleshooting</strong></summary>

### Common Issues

#### API Server Won't Start
```bash
# Check if port 8000 is in use
netstat -an | findstr :8000  # Windows
lsof -i :8000                # macOS/Linux

# Kill existing process if needed
taskkill /F /PID <PID>       # Windows
kill -9 <PID>               # macOS/Linux
```

#### Model File Not Found
```bash
# Verify model file exists
ls -la models/discount_eligibility_model.pkl

# If missing, contact repository maintainer for model file
```

#### Streamlit Command Not Found
```bash
# Install streamlit
pip install streamlit

# Use python module syntax
python -m streamlit run src/dashboard.py
```

#### API Connection Errors
1. Verify API server is running on port 8000
2. Check `.env` file has correct API_KEY
3. Ensure firewall isn't blocking connections
4. Try accessing http://localhost:8000/docs directly

#### Dashboard Not Loading
1. Check Streamlit is installed: `streamlit version`
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Try clearing cache: Delete `.streamlit` folder
4. Check Python version compatibility (3.8+)

### Performance Issues
- **Slow API Response**: Check model file size and available RAM
- **Dashboard Lag**: Disable auto-refresh features
- **High Memory Usage**: Restart services periodically during development

### Getting Help
- Check the API documentation at http://localhost:8000/docs
- Review error logs in the terminal output
- Ensure all environment variables are properly set
- Verify Python package versions match requirements.txt

</details>

<details>
<summary><strong>🚀 Deployment</strong></summary>

### Production Deployment

#### Environment Setup
```bash
# Production environment variables
API_KEY=<strong-random-key>
POLICY_SYSTEM_URL=<production-policy-api>
POLICY_SYSTEM_API_KEY=<production-policy-key>
LOG_LEVEL=INFO
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY .env .

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Cloud Deployment Options
- **AWS**: ECS with Application Load Balancer
- **Google Cloud**: Cloud Run with Cloud SQL
- **Azure**: Container Instances with App Gateway
- **Heroku**: Container deployment with add-ons

#### Scaling Considerations
- Use Redis for session management
- Implement connection pooling for database
- Set up horizontal pod autoscaling
- Configure health checks and monitoring

</details>

## Business Value

**Customer Benefits:**
- Up to 30% premium savings for safe drivers
- Transparent, behavior-based pricing
- Real-time premium adjustments
- Incentivized safe driving

**Business Benefits:**
- 90% reduction in manual processing
- Improved risk assessment accuracy
- Enhanced customer retention
- Competitive market differentiation

## Technical Highlights

- **Real-time Processing**: Sub-200ms API response times
- **Production Ready**: Comprehensive error handling and monitoring
- **Scalable Architecture**: RESTful APIs with async processing
- **Enterprise Integration**: Batch processing and webhook support
- **Security First**: API key authentication and audit trails

## Scalable cloud scope : 
1. Data Ingestion and Storage:
The first step is to replace local CSV files with a scalable Amazon S3 data lake for central storage, using a real-time service like Amazon Kinesis for data ingestion. This approach creates a durable and infinitely scalable foundation capable of handling continuous data streams from millions of vehicles. My data_simulator.py script is valuable here, as it defines the precise data schema that the new real-time ingestion pipeline will expect and process.

2. Data Processing and Feature Engineering 
To overcome the limitations of single-machine processing, the feature engineering workflow will be migrated from pandas to a distributed engine like Apache Spark running on AWS Glue. This serverless approach allows for the efficient processing of terabytes of data by automatically scaling compute resources as needed. My feature_engineering.py script serves as the perfect logical blueprint, as its aggregation and transformation rules will be directly translated into the new, highly scalable PySpark job.

3. Model Training and Management
Model training will be moved from a local machine to Amazon SageMaker, a dedicated ML platform. This provides access to powerful, on-demand compute resources (including GPUs), enabling me to train more complex models on massive datasets far more quickly. SageMaker also offers robust experiment tracking to systematically log and compare model performance. The core model definition and training logic within my train_model.py can be easily adapted to run as a scalable SageMaker training job.

4. Model Deployment and Serving
To ensure the prediction service is robust and scalable, my FastAPI application will be packaged into a Docker container and deployed on a managed service like Amazon SageMaker Endpoints. This creates a highly available API that automatically scales to handle traffic fluctuations, ensuring low-latency predictions for millions of users. My api.py script is essential as it provides the complete application logic, including endpoints and data validation, which will be placed directly into the container for deployment.

5. Orchestration and MLOps
Finally, the entire process will be automated by connecting each stage into a cohesive workflow using a service like AWS Step Functions. This creates a hands-off MLOps pipeline that automatically triggers feature engineering, model retraining, and deployment as new data becomes available. The sequential order in which I currently run my scripts serves as the exact blueprint for this automated workflow, ensuring a reliable and continuously improving system without manual intervention.



## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


