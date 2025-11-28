import os
import traceback
from pathlib import Path
import sqlite3
import pandas as pd


def create_db_from_csv(db_path='student_performance.db', csv_path='data/raw_student_data.csv'):
    """Create a simple SQLite database from the provided CSV file.
    Writes a 'scores' table. Returns (success:bool, message:str).
    """
    try:
        if not os.path.exists(csv_path):
            return False, f'CSV not found: {csv_path}'
        df_csv = pd.read_csv(csv_path)
        conn = sqlite3.connect(db_path)
        df_csv.to_sql('scores', conn, if_exists='replace', index=False)
        conn.close()
        return True, 'Database created from CSV.'
    except Exception as e:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'db_errors.log'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n--- CREATE DB ERROR ---\n')
            f.write(traceback.format_exc())
        return False, str(e)
