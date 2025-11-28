import sqlite3
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Configuration
DB_PATH = 'student_performance.db'
MODEL_DIR = 'models'

def train_models():
    print("Training ML models...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load data
    # We want to predict total_mark based on attendance and individual exams (simulating predicting final from partials)
    # Or predict performance category based on demographics and attendance
    query = """
    SELECT 
        sc.total_attendance,
        sc.exam_1,
        sc.exam_2,
        sc.exam_3,
        sc.total_mark,
        st.gender,
        s.subject_name
    FROM scores sc
    JOIN students st ON sc.student_id = st.student_id
    JOIN subjects s ON sc.subject_id = s.subject_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No data found!")
        return

    # Preprocessing
    # Convert categorical to numeric
    df['gender_code'] = df['gender'].map({'Male': 0, 'Female': 1})
    # One-hot encode subjects? For simplicity, let's just use numeric features for regression
    
    # --- Model 1: Regression (Predict Total Mark) ---
    # Features: Attendance, Exam 1, Exam 2 (Predicting Final based on partials + attendance)
    X_reg = df[['total_attendance', 'exam_1', 'exam_2']]
    y_reg = df['total_mark']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = LinearRegression()
    reg_model.fit(X_train_r, y_train_r)
    
    y_pred_r = reg_model.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    print(f"Regression Model MSE: {mse:.2f}")
    
    # Save Regression Model
    with open(os.path.join(MODEL_DIR, 'score_predictor.pkl'), 'wb') as f:
        pickle.dump(reg_model, f)
        
    # --- Model 2: Classification (Predict Performance Group) ---
    # Low (<70), Medium (70-85), High (>85)
    def categorize(score):
        if score < 70: return 'Low'
        elif score < 85: return 'Medium'
        else: return 'High'
        
    df['performance_group'] = df['total_mark'].apply(categorize)
    
    # Features: Attendance, Gender Code
    X_clf = df[['total_attendance', 'gender_code', 'exam_1']]
    y_clf = df['performance_group']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    
    y_pred_c = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"Classification Model Accuracy: {acc:.2f}")
    print(classification_report(y_test_c, y_pred_c))
    
    # Save Classification Model
    with open(os.path.join(MODEL_DIR, 'performance_classifier.pkl'), 'wb') as f:
        pickle.dump(clf_model, f)

    print("Models saved.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    train_models()
