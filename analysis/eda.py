import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DB_PATH = 'student_performance.db'
OUTPUT_DIR = 'docs/images'

def run_eda():
    print("Starting Deep Analysis EDA...")
    conn = sqlite3.connect(DB_PATH)
    
    # Set style
    plt.style.use('dark_background')
    sns.set_palette("bright")

    # 1. Subject Performance Gap (Heatmap)
    print("Generating Performance Gap Heatmap...")
    query_gap = """
    SELECT 
        strftime('%Y-%m', sc.exam_date) as month,
        s.subject_name,
        AVG(sc.total_mark) - (SELECT AVG(total_mark) FROM scores s2 
                              WHERE strftime('%Y-%m', s2.exam_date) = strftime('%Y-%m', sc.exam_date)) as performance_gap
    FROM scores sc
    JOIN subjects s ON sc.subject_id = s.subject_id
    GROUP BY month, s.subject_name
    ORDER BY month;
    """
    df_gap = pd.read_sql_query(query_gap, conn)
    
    if not df_gap.empty:
        pivot_gap = df_gap.pivot(index='subject_name', columns='month', values='performance_gap')
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_gap, cmap='coolwarm', center=0, annot=False)
        plt.title('Subject Performance Gap vs Monthly Average', fontsize=16, color='white')
        plt.xlabel('Month')
        plt.ylabel('Subject')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'performance_gap_heatmap.png'))
        plt.close()

    # 2. Attendance Volatility
    print("Generating Attendance Volatility Chart...")
    query_volatility = """
    SELECT 
        strftime('%Y-%m', exam_date) as month,
        (MAX(total_attendance) - MIN(total_attendance)) as attendance_variance,
        AVG(total_attendance) as avg_attendance
    FROM scores
    GROUP BY month
    ORDER BY month;
    """
    df_vol = pd.read_sql_query(query_volatility, conn)
    
    if not df_vol.empty:
        plt.figure(figsize=(12, 6))
        # Dual axis
        ax1 = sns.barplot(data=df_vol, x='month', y='attendance_variance', color='#b537f2', alpha=0.6)
        ax2 = ax1.twinx()
        sns.lineplot(data=df_vol, x='month', y='avg_attendance', ax=ax2, color='#00f5ff', marker='o')
        
        ax1.set_ylabel('Attendance Variance (Range)', color='#b537f2')
        ax2.set_ylabel('Average Attendance', color='#00f5ff')
        plt.title('Monthly Attendance: Average vs Volatility', fontsize=16, color='white')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'attendance_volatility.png'))
        plt.close()

    # 3. Teacher Value Added (Scatter)
    print("Generating Teacher Value Added Chart...")
    query_value = """
    SELECT 
        t.first_name || ' ' || t.last_name as teacher_name,
        s.subject_name,
        AVG(sc.total_mark) - (SELECT AVG(total_mark) FROM scores s2 WHERE s2.subject_id = sc.subject_id) as value_added
    FROM scores sc
    JOIN teachers t ON sc.teacher_id = t.teacher_id
    JOIN subjects s ON sc.subject_id = s.subject_id
    GROUP BY t.teacher_id, s.subject_name
    ORDER BY value_added DESC;
    """
    df_value = pd.read_sql_query(query_value, conn)
    
    if not df_value.empty:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_value, x='value_added', y='teacher_name', hue='subject_name', s=100)
        plt.axvline(0, color='white', linestyle='--', alpha=0.5)
        plt.title('Teacher Value Added (Score vs Subject Avg)', fontsize=16, color='white')
        plt.xlabel('Value Added (Points above/below avg)', fontsize=12)
        plt.ylabel('Teacher', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'teacher_value_added.png'))
        plt.close()

    conn.close()
    print("Deep Analysis EDA Complete.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    run_eda()
