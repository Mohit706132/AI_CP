import sqlite3
from datetime import datetime
from pathlib import Path
import json

DB_PATH = Path(__file__).resolve().parent.parent / "audit.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            username TEXT,
            module TEXT,
            sample_name TEXT,
            prediction TEXT,
            status TEXT,
            purity REAL,
            confidence_json TEXT,
            modality TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_result(username, module, result_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO audit_log (timestamp, username, module, sample_name, prediction, status, purity, confidence_json, modality)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        username,
        module,
        result_dict.get('sample_name', 'Unknown'),
        result_dict.get('prediction', 'Unknown'),
        result_dict.get('adul_status', 'Unknown'),
        result_dict.get('purity', 0.0),
        json.dumps(result_dict.get('confidence', {})),
        result_dict.get('modality', 'Unknown')
    ))
    conn.commit()
    conn.close()

def log_batch_results(username, module, results):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = []
    for r in results:
        data.append((
            now, username, module,
            r.get('sample_name', 'Unknown'),
            r.get('prediction', 'Unknown'),
            r.get('adul_status', 'Unknown'),
            r.get('purity', 0.0),
            json.dumps(r.get('confidence', {})),
            r.get('modality', 'Unknown')
        ))
        
    c.executemany('''
        INSERT INTO audit_log (timestamp, username, module, sample_name, prediction, status, purity, confidence_json, modality)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def get_audit_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM audit_log ORDER BY id DESC LIMIT 1000')
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]
