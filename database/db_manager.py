import sqlite3
import pandas as pd
import random
from faker import Faker
import os
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Configuration
DB_PATH = 'student_performance.db'
SCHEMA_PATH = 'database/schema.sql'
RAW_DATA_PATH = 'data/raw_student_data.csv'

def init_db():
    """Initialize the database with the schema."""
    print("Initializing database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()
        cursor.executescript(schema)
    
    conn.commit()
    conn.close()
    print("Database initialized.")

def generate_data():
    """Generate synthetic data and populate the database."""
    print("Generating data...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Generate Subjects
    subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'History', 'English', 'Computer Science', 'Art']
    subject_ids = []
    for sub in subjects:
        sub_id = f"SUB_{sub[:3].upper()}_{random.randint(100, 999)}"
        cursor.execute("INSERT OR IGNORE INTO subjects (subject_id, subject_name) VALUES (?, ?)", (sub_id, sub))
        subject_ids.append(sub_id)
    
    # 2. Generate Teachers
    teacher_ids = []
    for _ in range(10):
        t_id = f"TCH_{fake.uuid4()[:8]}"
        cursor.execute("""
            INSERT INTO teachers (teacher_id, first_name, last_name, gender, subject, date_of_employment, phone_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            t_id,
            fake.first_name(),
            fake.last_name(),
            random.choice(['Male', 'Female']),
            random.choice(subjects),
            fake.date_between(start_date='-10y', end_date='today'),
            fake.phone_number()
        ))
        teacher_ids.append(t_id)

    # 3. Generate Students
    student_ids = []
    for _ in range(500):
        s_id = f"STU_{fake.uuid4()[:8]}"
        cursor.execute("""
            INSERT INTO students (student_id, first_name, last_name, gender, class, date_of_birth)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            s_id,
            fake.first_name(),
            fake.last_name(),
            random.choice(['Male', 'Female']),
            f"Class {random.randint(10, 12)}",
            fake.date_of_birth(minimum_age=15, maximum_age=18)
        ))
        student_ids.append(s_id)

    # 4. Generate Scores
    # Each student takes 3-5 subjects
    score_data = []
    
    for s_id in student_ids:
        num_subjects = random.randint(3, 5)
        chosen_subjects = random.sample(subject_ids, num_subjects)
        
        for sub_id in chosen_subjects:
            score_id = f"SCR_{fake.uuid4()[:8]}"
            t_id = random.choice(teacher_ids) # Assign random teacher for now
            
            # Generate exam scores
            exam_1 = round(random.uniform(50, 100), 2)
            exam_2 = round(random.uniform(50, 100), 2)
            exam_3 = round(random.uniform(50, 100), 2)
            
            # Generate attendance (lectures)
            lectures = [random.choice([True, False]) for _ in range(10)]
            # Bias towards True for better realistic data
            lectures = [random.random() > 0.2 for _ in range(10)] 
            
            total_attendance = sum(lectures)
            
            # Calculate total mark (simple average for now)
            total_mark = round((exam_1 + exam_2 + exam_3) / 3, 2)

            cursor.execute("""
                INSERT INTO scores (
                    score_id, student_id, teacher_id, subject_id, 
                    exam_1, exam_2, exam_3, 
                    lecture_1, lecture_2, lecture_3, lecture_4, lecture_5, 
                    lecture_6, lecture_7, lecture_8, lecture_9, lecture_10, 
                    total_attendance, total_mark,
                    exam_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score_id, s_id, t_id, sub_id,
                exam_1, exam_2, exam_3,
                *lectures,
                total_attendance, total_mark,
                fake.date_between(start_date='-2y', end_date='today')
            ))
            
            # Collect data for CSV export
            score_data.append({
                'student_id': s_id,
                'subject_id': sub_id,
                'teacher_id': t_id,
                'exam_1': exam_1,
                'exam_2': exam_2,
                'exam_3': exam_3,
                'attendance': total_attendance,
                'final_score': total_mark,
                'date': fake.date_this_year()
            })

    conn.commit()
    conn.close()
    print("Data generation complete.")
    
    # Export to CSV
    df = pd.DataFrame(score_data)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw data exported to {RAW_DATA_PATH}")

if __name__ == "__main__":
    init_db()
    generate_data()
