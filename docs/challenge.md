# Software Engineer (ML & LLMs) Challenge

- [Overview](#overview)
- [Part I: Model](#part-i-model)
    - [Analysis](#analysis)
    - [Model Transcription](#model-transcription)
    - [Tests](#tests)
- [Part II: FastAPI App](#part-ii-fastapi-app)
    - [App Creation](#app-creation)
    - [Tests](#tests)
- [Part III: Deployment to GCP](#part-iii-deployment-to-gcp)
    - [Deployment](#deployment)
    - [Tests](#tests)
- [Part IV: CI/CD & Git](#part-iv-cicd--git)
    - [CI/CD Pipeline](#cicd-pipeline)
    - [Continuous Integration (CI)](#continuous-integration-ci)
    - [Continuous Delivery (CD)](#continuous-delivery-cd)
    - [Git](#git)

## Overview

This document explains the results of the execution of this challenge, including: 
- Mayor decisions taken
- Steps taken
- Analysis made
- Others.

It is divided into 4 Parts, mimicking the instructions for the challenge:
- Part I: Model
- Part II: FastAPI app
- Part III: Deployment to GCP
- Part IV: CI/CD & Git

## Part I: Model

### Analysis

A short analysis on the resulting model was made:

- The only features used were OPERA, TIPOVUELO, MES, which seems to have been a mistake; considering the plots, there should be useful information that can be added to the model
- A proper validation dataset was not used, which makes the found metrics for a test dataset not fully valid as performance metrics, as they were used for taking desicions on the model
- The train and test split was done without considering the time relation between the samples, making the metrics less representative for a real production setting; and not forcing the same class distribution (even though it ended up not being too different), which can be worth fixing for consistency in future cases.
- Feature importance was taken from a model predicting everything as class 0, meaning that it didn't learn useful information from the data, that makes the listed feature importance not useful, including the top 10 features
- It is unclear which metric is the one being used to select the model, recall for class 1 is mentioned, but could fall short in considering false positives; it's worth noting that a dummy model predicting everything as 0 has a high weighted f1-score and accuracy
- Training the models without feature importance and with balance, made the Xgboost model better than before based on recall for class 1 (better than with feature importance selection and balance)
- It's clear that xgboost could be fine-tuned more to achieve better results (or try other models, other feature extraction, etc), but since it's not the scope of the challenge, it'll be ignored

**The newly added xgboost model (in this analysis) is the model that has the best recall for class 1, and does not worsen the f1 score. But to make sure to make the model compatible with the tests in the challenge, and to follow the DS conclusions, the logistic model with the top 10 features was used for production. Ultimately, if the model is useful at all depends on whether the benefit of predicting some delays outweighs the cost of having so many false positives. It's important to note that the model could be greatly improved by reworking the feature choice and extraction**


### Model Transcription

Relevant points: 
- The logistic regression model with balanced class weight and the top 10 features was used, following the DS conclusions (see analysis above).
- The one hot encoding was handled with "get_dummies" and setting the output features to the selected top 10 features, guaranteeing the correct feature shape and order for any input (does not require a trainable OneHotEncoder).
- File train.py was added to train and save the model, providing repeatability for future model trainings. Can be run with "make train".
- Model was saved to model.joblib file with build metadata in model_metadata.json for traceability, and is automatically loaded when creating a instance of DelayModel.
- Check for correct and sufficient column in data was added to the preprocess, raises custom exception InputDataException if conditions are not met.
- Warning is raised if model is not able to be loaded at runtime (can be trained at runtime by calling fit()), and exception is raised if prediction is called with no model loaded / trained.
- Logging was added accros the entire process.

### Tests

All 4 tests in test_model were passed

## Part II: FastAPI App

### App Creation

Relevant points:
- DelayModel was integrated to predict on data recieved on the endpoint /predict, it is loaded once at import time.
- Pydantic models for input and outdata were created (FlightData, PredictRequest, PredictResponse), providing both shape and content data validation.
- If the input data passed to the endpoint does not have the propper format or data, the exception RequestValidationError is raised by pydantic, returning a message with the formatted problem detail and a 400 http code.
- If ModelNotLoadedException is raised by the model on prediction, it is answered with a 503 service unavailable (model must be already trained to predict on data).
- Logging was configured across the api.

### Tests

All 4 tests in test_api were passed

Also, prediction test (test_should_get_predict) was modified to decouple it from model tests (and not depend on model specific predictions), by following the challenge comment left there, to mock the model prediction. 

## Part III: Deployment to GCP

### Deployment

App URL: https://latam-mle-app-2026-712938310629.us-central1.run.app

Docker image containing the app is sent to GCP Artifact Registry, and run in GCP Cloud Run; being publicly available to be consumed

Relevant points:
- Base image set to python:3.11-slim, same version used for development and testing; python:latest was avoided for reproducibility.
- Dependencies are installed before copying the application code, dependency layer is reused whenever only source files change.
- The container is compatible with Cloud Run's injected $PORT, defaulting to 8000 for local runs.
- A .dockerignore was added, reducing the build context by excluding unnecesary things like .venv, .git, data and test artifacts.
- Version for pandas was updated to the closest version with a wheel, all tests continue passing with no changes
- Makefile was updated to run stress test against the deployed app.

### Tests

The stress test was run against the deployed app using the parameters provided by the challenge Makefile (100 users, spawn rate 1/s, 60s run time):

- 5196 requests, 0 failures (0.00%)
- Median 330 ms, p95 590 ms, p99 700 ms, max 1217 ms
- ~113 req/s at peak

## Part IV: CI/CD & Git

### CI/CD Pipeline

Due to Github limitations, it was decided to combine CI and CD in a single workflow split into two jobs, instead of two files; because the keyword 'needs:' only works between jobs in the same workflow; separating is posible, but more complex and less cleanly traceable.

Relevant points:
- Direct pushes to main were blocked for safety, so CI/CD run only on a merge push.
- CI job runs on pull requests and push to dev and main (merge).
- CD job only runs if CI is successfull and restricted to run on push to main (merge).
- Linting with ruff was added to CI job to provide code quality checks. Limited to challenge/ by ruff.toml
- Reports are uploaded as artifacts with `if: always()`, so also available on failure.
- Images are tagged `run_id-sha`, so each deployment maps to one commit and one run.
- After deploying, the app service is checked calling /health before running the stress tests
- Authentication uses service account key. WIF is the standard, but its setup is not quite justified for a 3 day deployment; the key will be revoked after the review.
- Pipeline was validated by temporarily running on the feature branch, the first run on main was not its first execution.

### Continuous Integration (CI)

CI is triggered on pull request to _main_ or _dev_, and on push to _dev_ or _main_ (merge). Executes the following steps:
1. Checkout Code
2. Install Python
3. Install Make
4. Install Python requirements
5. Lint challenge folder with ruff
6. Run Model & API tests (any failure stops the pipeline, on every branch)
7. Upload Test Reports as an artifact

### Continuous Delivery (CD)

CD is triggered only on push to _main_ (merge), and only if CI succeeded. Executes the following steps:
1. Setup Env variables
2. Checkout Code
3. Install Python
4. Install Make
5. Install Python requirements
6. Setup GCP SDK with credentials
7. Setup CLI and configure project
8. Build Docker image
9. Push Docker image to GCP Artifact Registry
10. Deploy Docker container to GCP Cloud Run
11. Verify the deployed service answers on /health
12. Run Stress Test
13. Upload Test Reports as an artifact

### Git

Branches are structure as:
- main
- dev
- feature_*:
    - feature_model
    - feature_fastapi
    - feature_deployment_gcp
    - feature_cicd

Code was worked on feature branches then merged to _dev_ through pull requests, after all work was finished it was merged to _main_

Relevant points:
- main is protected: no direct pushes, merging requires a pull request.
- dev integrates the parts before they reach main, which stays deployable.
- Commit messages follow Conventional Commits with a scope.
- The final state is tagged v1.0.0 and published as a GitHub release.
- Branches are kept after merging, so they remain reviewable.