-- SQL Queries for Student Performance Dashboard
-- This file contains the SQL logic powering the dashboard's visualizations and insights.

-- ==========================================
-- PART 1: Attendance Trends
-- ==========================================

-- 1.1 Student Attendance vs Class Average (Insight)
-- VISUALIZATION: None (Used for identifying at-risk students in "Raw Data" or alerts)
-- INSIGHT: Identifies students who are significantly below their class average attendance.
SELECT 
    st.first_name || ' ' || st.last_name as student_name,
    st.class,
    sc.total_attendance,
    (SELECT AVG(total_attendance) FROM scores s2 
     JOIN students st2 ON s2.student_id = st2.student_id 
     WHERE st2.class = st.class) as class_avg_attendance,
    sc.total_attendance - (SELECT AVG(total_attendance) FROM scores s2 
                           JOIN students st2 ON s2.student_id = st2.student_id 
                           WHERE st2.class = st.class) as diff_from_class_avg
FROM scores sc
JOIN students st ON sc.student_id = st.student_id
WHERE diff_from_class_avg < -2 
ORDER BY diff_from_class_avg ASC;

-- 1.2 Monthly Attendance Volatility (Visualization)
-- VISUALIZATION: "Monthly Attendance Variance" Bar Chart (Advanced Insights Page)
-- INSIGHT: Shows which months had the most unstable attendance (High Variance = Unstable).
SELECT 
    month,
    avg_attendance,
    attendance_variance
FROM (
    SELECT 
        strftime('%Y-%m', exam_date) as month,
        AVG(total_attendance) as avg_attendance,
        (MAX(total_attendance) - MIN(total_attendance)) as attendance_variance
    FROM scores
    GROUP BY month
) as MonthlyStats
ORDER BY attendance_variance DESC;

-- ==========================================
-- PART 2: Monthly Performance
-- ==========================================

-- 2.1 Subject Performance vs Monthly Global Average (Visualization)
-- VISUALIZATION: "Subject Performance vs Global Average" Heatmap (Advanced Insights Page)
-- INSIGHT: Reveals if a specific subject was harder or easier than the school average for a given month.
SELECT 
    strftime('%Y-%m', sc.exam_date) as month,
    s.subject_name,
    AVG(sc.total_mark) as subject_monthly_avg,
    (SELECT AVG(total_mark) FROM scores s2 
     WHERE strftime('%Y-%m', s2.exam_date) = strftime('%Y-%m', sc.exam_date)) as global_monthly_avg,
    AVG(sc.total_mark) - (SELECT AVG(total_mark) FROM scores s2 
                          WHERE strftime('%Y-%m', s2.exam_date) = strftime('%Y-%m', sc.exam_date)) as performance_gap
FROM scores sc
JOIN subjects s ON sc.subject_id = s.subject_id
GROUP BY month, s.subject_name
HAVING performance_gap > 5 OR performance_gap < -5 
ORDER BY month, performance_gap DESC;

-- 2.2 Consistent High Performers (Insight)
-- VISUALIZATION: "Top Performers" Table (Overview Page)
-- INSIGHT: Lists students who maintained a score above 85 in ALL exams.
SELECT 
    st.first_name || ' ' || st.last_name as student_name,
    st.class
FROM students st
WHERE st.student_id IN (
    SELECT student_id 
    FROM scores 
    GROUP BY student_id 
    HAVING MIN(total_mark) > 85
);

-- ==========================================
-- PART 3: Teacher Impact
-- ==========================================

-- 3.1 Teacher Impact on "At-Risk" Students (Visualization)
-- VISUALIZATION: "Best Teachers for At-Risk Students" Bar Chart (Advanced Insights Page)
-- INSIGHT: Highlights teachers who achieve high scores even with students who have low attendance (<5).
SELECT 
    t.first_name || ' ' || t.last_name as teacher_name,
    COUNT(sc.score_id) as at_risk_students_taught,
    AVG(sc.total_mark) as avg_score_for_at_risk
FROM scores sc
JOIN teachers t ON sc.teacher_id = t.teacher_id
WHERE sc.student_id IN (
    SELECT student_id FROM scores WHERE total_attendance < 5
)
GROUP BY t.teacher_id
HAVING at_risk_students_taught > 5
ORDER BY avg_score_for_at_risk DESC;

-- 3.2 Teacher Value Added (Subquery)
-- VISUALIZATION: "Teacher Value Added" Scatter Plot (Advanced Insights Page)
-- INSIGHT: Compares a teacher's average student score against the global average for that subject.
-- Positive "Value Added" means their students perform better than the subject norm.
SELECT 
    t.first_name || ' ' || t.last_name as teacher_name,
    s.subject_name,
    AVG(sc.total_mark) as teacher_avg_score,
    (SELECT AVG(total_mark) FROM scores s2 WHERE s2.subject_id = sc.subject_id) as subject_global_avg,
    AVG(sc.total_mark) - (SELECT AVG(total_mark) FROM scores s2 WHERE s2.subject_id = sc.subject_id) as value_added
FROM scores sc
JOIN teachers t ON sc.teacher_id = t.teacher_id
JOIN subjects s ON sc.subject_id = s.subject_id
GROUP BY t.teacher_id, s.subject_name
ORDER BY value_added DESC;

-- 3.3 Teacher Consistency (Variance Analysis)
-- VISUALIZATION: "Teacher Consistency" Table (Advanced Insights Page)
-- INSIGHT: Measures the standard deviation of scores. Lower deviation means the teacher produces consistent results across all students.
-- Note: SQLite doesn't have a native STDEV function by default in all versions, so we approximate variance or use a complex calculation.
-- Here we calculate the range (Max - Min) and Average Deviation as proxies for consistency.
SELECT 
    t.first_name || ' ' || t.last_name as teacher_name,
    COUNT(sc.score_id) as students_taught,
    AVG(sc.total_mark) as avg_score,
    (MAX(sc.total_mark) - MIN(sc.total_mark)) as score_range,
    AVG(ABS(sc.total_mark - (SELECT AVG(total_mark) FROM scores s2 WHERE s2.teacher_id = t.teacher_id))) as avg_deviation
FROM scores sc
JOIN teachers t ON sc.teacher_id = t.teacher_id
GROUP BY t.teacher_id
HAVING students_taught > 10
ORDER BY avg_deviation ASC; -- Ascending because lower deviation = higher consistency
