# Student Performance Dashboard

## 📊 Overview
This project is a comprehensive data pipeline and analytics dashboard designed to track and analyze student academic performance. It demonstrates the integration of SQL databases, Python data processing, Machine Learning, and a modern web interface.

The system simulates a real-world educational environment with **500 students**, tracking their performance across **2 years** of exams and attendance records.

## ✨ Key Features

### 1. Advanced Data Pipeline
- **Synthetic Data Generation**: Uses `Faker` to generate realistic data for Students, Teachers, Subjects, and Scores.
- **Scalable Database**: Normalized SQLite schema (`student_performance.db`) optimized for complex queries.
- **Data Expansion**: Includes a dataset of 500 students with temporal data spanning 2 years.

### 2. Deep Analytics & Insights
The dashboard provides nuanced insights derived from complex SQL queries (Subqueries, CTEs, Window Functions):
- **Attendance Volatility**: Identifies months with unstable attendance patterns.
- **Performance Gaps**: Heatmap visualization showing how specific subjects perform relative to the global monthly average.
- **Teacher Impact**: Analysis of which teachers are most effective with "at-risk" students (low attendance).
- **Monthly Growth**: Time-series analysis of student performance trends.

### 3. Machine Learning Integration
- **Score Prediction (Regression)**: Predicts a student's final score based on attendance and partial exam results.
- **Performance Classification**: Categorizes students into High/Medium/Low performance groups.

### 4. Modern Web Dashboard
- **Streamlit Application**: A high-fidelity, interactive dashboard.
- **"Neon Dark" Aesthetic**: Custom CSS styling matching modern UI design principles.
- **Interactive Visualizations**: Powered by Plotly and Seaborn.

## 📂 Directory Structure
```
student_dashboard/
├── app/
│   ├── main.py         # Main Streamlit application
│   └── styles.py       # Custom CSS and UI components
├── database/
│   ├── schema.sql      # Database schema definition
│   ├── queries.sql     # SQL queries for insights & visualizations
│   └── db_manager.py   # Data generation and seeding script
├── models/
│   ├── train_model.py  # ML model training script
│   └── *.pkl           # Saved ML models
├── analysis/
│   └── eda.py          # Exploratory Data Analysis script
├── data/
│   └── raw_student_data.csv  # Exported raw data
├── docs/               # Generated plots and documentation
└── requirements.txt    # Project dependencies
```

## 🚀 Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Initialize Database & Generate Data**:
    ```bash
    python database/db_manager.py
    ```

3.  **Train ML Models**:
    ```bash
    python models/train_model.py
    ```

4.  **Run Dashboard**:
    ```bash
    streamlit run app/main.py
    ```

## 🛠️ Technologies
- **Language**: Python 3.x
- **Database**: SQLite
- **Libraries**: Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly, Faker
- **Web Framework**: Streamlit

## 📈 Visualizations Included
- **Score Distribution by Gender** (Histogram)
- **Attendance vs Score Correlation** (Scatter Plot)
- **Subject Performance Heatmap** (Advanced)
- **Monthly Attendance Variance** (Bar Chart)
- **Teacher Effectiveness for At-Risk Students** (Bar Chart)
