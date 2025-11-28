-- Schema for Student Performance Dashboard

CREATE TABLE IF NOT EXISTS teachers (
    teacher_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    gender VARCHAR(10),
    subject VARCHAR(50),
    date_of_employment DATE,
    phone_number VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id VARCHAR(50) PRIMARY KEY,
    subject_name VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    gender VARCHAR(10),
    class VARCHAR(20),
    date_of_birth DATE
);

CREATE TABLE IF NOT EXISTS scores (
    score_id VARCHAR(50) PRIMARY KEY,
    student_id VARCHAR(50),
    teacher_id VARCHAR(50),
    subject_id VARCHAR(50),
    exam_1 FLOAT,
    exam_2 FLOAT,
    exam_3 FLOAT,
    lecture_1 BOOLEAN,
    lecture_2 BOOLEAN,
    lecture_3 BOOLEAN,
    lecture_4 BOOLEAN,
    lecture_5 BOOLEAN,
    lecture_6 BOOLEAN,
    lecture_7 BOOLEAN,
    lecture_8 BOOLEAN,
    lecture_9 BOOLEAN,
    lecture_10 BOOLEAN,
    total_attendance INTEGER,
    total_mark FLOAT,
    exam_date DATE,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);
