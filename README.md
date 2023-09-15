# PotatoDiseaseClassifier: End-to-End Deep Learning Project

## Overview

This repository contains the code and documentation for an end-to-end deep learning project for potato disease classification. The project includes data preprocessing, model building, deployment, model version management, and deployment on Google Cloud Platform (GCP). Additionally, the project showcases model quantization for mobile deployment using TensorFlow Lite.

## Features

- Data Cleaning: The project starts with cleaning and preprocessing the potato disease image dataset.

- Data Augmentation: Data augmentation techniques are applied to increase the diversity of the training dataset.

- Model Building: Deep learning models are developed to classify potato disease based on input images.

- FastAPI Deployment: A FastAPI application is created for serving the trained model via RESTful APIs.

- TensorFlow Serving: TensorFlow Serving is used for model version management and scalable deployment.

- Google Cloud Platform Deployment: The project demonstrates how to deploy models and APIs on GCP for high availability.

- Model Quantization: Model quantization is applied to convert the deep learning model into a lightweight format for mobile deployment.

- Flutter App: A Flutter mobile app is developed to showcase the mobile deployment of the potato disease classifier.
## Usage
Data Cleaning and Augmentation: Use Jupyter notebooks or scripts in the data_preprocessing directory to preprocess and augment your dataset.

Model Training: Train your deep learning models using the scripts in the model_training directory. Save the trained models in the saved_model directory.

FastAPI Deployment: Start the FastAPI server by running fastapi_app.py in the deployment directory. Access the API at http://localhost:8000/docs.

TensorFlow Serving: Deploy models using TensorFlow Serving on your GCP instance following the guide in the tensorflow_serving directory.

Mobile Deployment: Use the TensorFlow Lite conversion script in the mobile_app directory to convert your model. Develop a Flutter app in the flutter_app directory for mobile deployment.

Acknowledgments
Special thanks to https://www.youtube.com/@codebasics  and the open-source community and TensorFlow for providing tools and resources for deep learning.
