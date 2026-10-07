
# 🏦 Loan Approval Prediction App

A machine learning web app that predicts whether a loan application is likely to be approved or rejected, built with **Streamlit** and a **Decision Tree** classifier.

🔗 **Live Demo:** [https://loan-prediction-apps-rdwrzbbmcvwcsnt5b6ryiy.streamlit.app](https://loan-prediction-apps-rdwrzbbmcvwcsnt5b6ryiy.streamlit.app)

## Overview

The user enters applicant details in a simple form, and the trained model returns an instant prediction along with the approval probability.

## Input Features

| Feature | Description |
|---|---|
| Gender | Male or Female |
| Married | Yes or No |
| Dependents | 0, 1, 2, 3+ |
| Education | Graduate or Not Graduate |
| Self Employed | Yes or No |
| Applicant Income | Monthly income of the applicant |
| Coapplicant Income | Monthly income of the coapplicant |
| Loan Amount | Loan amount (in thousands) |
| Loan Amount Term | Loan duration in months |
| Credit History | Good (1) or Bad (0) |
| Property Area | Rural, Semiurban or Urban |

## Model

- **Algorithm:** Decision Tree Classifier (`max_depth=2`)
- **Target:** Loan status (`Y` = approved, `N` = rejected)
- **Most important features:** Credit History (about 93 percent) and Coapplicant Income (about 7 percent)

## Project Structure

```
├── app.py                # Streamlit application
├── final_model.pkl       # Trained Decision Tree model
├── label_encoder.pkl     # Label encoder for the target (Y, N)
├── columns.pkl           # Expected feature
