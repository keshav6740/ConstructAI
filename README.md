# 🏗️ AI-Powered Construction Project Management System

## 📌 Overview

This project leverages AI, ML, and automation to optimize project execution in large-scale construction. It predicts delays, cost overruns, and assesses potential risks, helping teams make data-driven decisions.

## 🚀 Features

- 📊 **Predictive Analytics**: AI-driven forecasts for project delays and cost overruns.
- ⚠️ **Risk Management**: Identifies potential risks and mitigation strategies.
- 🔍 **Real-time Monitoring**: Tracks workforce and resource utilization.
- 📡 **FastAPI Backend**: High-performance API for managing project data.
- 💾 **PostgreSQL Database**: Stores project details, predictions, and risks.

## 🛠️ Tech Stack

- **Backend**: FastAPI, SQLAlchemy
- **Database**: PostgreSQL
- **Machine Learning**: Scikit-learn, Joblib (Pre-trained models)
- **Deployment**: Uvicorn

## 📂 Project Structure

```
├── main.py                 # FastAPI backend
├── models/                 # ML models for predictions
├── database/               # PostgreSQL integration
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

## 🏗️ Setup & Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/keshav6740/ConstructAi.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ConstructionAI
   ```
3. Create a virtual environment:
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
5. Start the FastAPI server:
   ```sh
   uvicorn main:app --reload
   ```

## 📡 API Endpoints

- `` → Create a new project
- `` → Retrieve all projects
- `` → Generate AI-based project predictions
- `` → Update actual delay and cost overrun

🚀 **Contribute & Star ⭐ the repo if you find it useful!**
