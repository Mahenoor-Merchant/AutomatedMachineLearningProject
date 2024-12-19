

# Automated Machine Learning Web Application

This is a **Flask-based web application** that automates machine learning workflows, including model training, analysis, prediction, and deployment to **AWS** or **GCP**. The application allows users to upload datasets, select a machine learning problem type (classification or regression), and receive a trained model along with predictions. Additionally, the trained models can be deployed to cloud platforms like AWS or GCP for production use.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [API Routes](#api-routes)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

---

## Features

- **Data Upload and Preprocessing**: Allows users to upload datasets for preprocessing.
- **Automated Model Training**: Supports training of classification and regression models with automated pipelines.
- **Model Prediction**: Makes predictions using the trained models and provides downloadable results.
- **Cloud Deployment**: Deploys models to **AWS** or **GCP** for production use.
- **Results Download**: Provides prediction results and data analysis in downloadable ZIP files.

---
## Requirements

- **Python 3.9.20**
- **Flask** (for the web framework)
- **AWS SDK (Boto3)** for interacting with AWS services
- **Google Cloud SDK** for GCP deployment
- **Additional Python libraries**: mentioned in requirements.txt
### Install Required Dependencies

1. Clone the repository:
   ```cmd
   git clone https://github.com/Mahenoor-Merchant/AutomatedMachineLearning.git
   ```

2. Set up a Python virtual environment:
   ```cmd
   conda create -p automlvenv python=3.9 -y
   ```

3. Activate the virtual environment:

     ```cmd
     conda activate automlvenv
     ```

4. Install the required packages:
   ```cmd
   python setup.py install
   ```

---

## Configuration

### AWS Credentials

To deploy the model on **AWS**, you'll need to configure your AWS credentials:

 **Using environment variables**: Set the following environment variables:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`



### GCP Credentials

To deploy the model on **GCP**, authenticate using the **Google Cloud SDK** or a **Service Account JSON** file.

1. Download your **GCP Service Account JSON** from the [Google Cloud Console](https://console.cloud.google.com/).
2. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
   ```

### Sensitive Variables

You can store sensitive variables such as **API keys** in a `.env` file. Make sure to add the `.env` file to `.gitignore` to prevent it from being committed to the repository.

Example `.env` file:
```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/gcp_credentials.json
```

---

## Running the Application

1. Activate the virtual environment (if not already activated).

2. Run the Flask application:
   ```cmd
   python app.py
   ```

3. The application will be accessible at `http://127.0.0.1:5000/` on your browser.

---

## API Routes

### 1. **Home Page (`/`)**
   - Displays a welcome message.

### 2. **Classification Prediction (`/predict_classification`)**
   - **Method**: `POST`
     - Upload a dataset, train a classification model, and receive predictions in a downloadable ZIP file containing:
       - Prediction results
       - Data analysis report
   - **Method**: `GET`
     - Displays a form for uploading classification data.

### 3. **Regression Prediction (`/predict_regression`)**
   - **Method**: `POST`
     - Upload a dataset, train a regression model, and receive predictions in a downloadable ZIP file containing:
       - Prediction results
       - Data analysis report
   - **Method**: `GET`
     - Displays a form for uploading regression data.

### 4. **Model Deployment (`/deploy`)**
   - **Method**: `POST`
     - Deploy the trained model on **AWS** or **GCP** by selecting the desired platform.
   - **Method**: `GET`
     - Displays a form for selecting the deployment platform (AWS/GCP).

---

## Troubleshooting

- **Issue**: Application is not running.
  - **Solution**: Ensure that all dependencies are installed and the virtual environment is activated.

- **Issue**: "Model not deployed" error.
  - **Solution**: Double-check AWS or GCP credentials and ensure that the environment variables are set correctly.

- **Issue**: Missing files after deployment.
  - **Solution**: Make sure the model and analysis files are properly generated before deploying.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

- This project uses **Flask** for the web framework.
- **Boto3** and **Google Cloud SDK** are used for cloud deployments.
- The machine learning workflows are built using **scikit-learn**, **pandas**, and other relevant libraries.

---
