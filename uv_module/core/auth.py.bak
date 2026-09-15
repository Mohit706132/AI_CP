# -*- coding: utf-8 -*-
"""Authentication and Role-Based Access Control (RBAC) module for Integrated Plant Suite."""

import os
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "audit.db"


def _hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    """Hashes password using PBKDF2-HMAC-SHA256 with salt."""
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return key.hex(), salt.hex()


def _verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return key.hex() == stored_hash


def init_auth_db():
    """Creates the users table and seeds a default admin account if not already present."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            approved_by TEXT,
            approved_at TEXT,
            last_login TEXT
        )
    ''')
    conn.commit()

    # Check if admin user exists
    c.execute("SELECT id FROM users WHERE role = 'admin'")
    if not c.fetchone():
        p_hash, salt = _hash_password("admin123")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''
            INSERT INTO users (username, full_name, password_hash, salt, role, status, created_at, approved_by, approved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ("admin", "System Administrator", p_hash, salt, "admin", "approved", now, "system", now))
        conn.commit()

    conn.close()


def register_user(username: str, password: str, full_name: str = "") -> tuple[bool, str]:
    """Registers a new user with status 'pending' awaiting Admin approval."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False, f"Username '{username}' is already registered."

    p_hash, salt = _hash_password(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        c.execute('''
            INSERT INTO users (username, full_name, password_hash, salt, role, status, created_at)
            VALUES (?, ?, ?, ?, 'user', 'pending', ?)
        ''', (username, full_name.strip() or username, p_hash, salt, now))
        conn.commit()
        conn.close()
        return True, "Registration submitted successfully! Your account is pending Admin approval."
    except Exception as e:
        conn.close()
        return False, f"Registration failed: {e}"


def verify_login(username: str, password: str) -> tuple[bool, str, dict | None]:
    """Verifies login credentials and checks user approval status."""
    username = username.strip().lower()
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()

    if not user:
        conn.close()
        return False, "Invalid username or password.", None

    user_dict = dict(user)
    if not _verify_password(password, user_dict["password_hash"], user_dict["salt"]):
        conn.close()
        return False, "Invalid username or password.", None

    status = user_dict.get("status", "pending")
    if status == "pending":
        conn.close()
        return False, "Your account registration has been submitted and is pending Admin approval.", None
    elif status == "rejected":
        conn.close()
        return False, "Your account access request was rejected by an Administrator.", None
    elif status != "approved":
        conn.close()
        return False, f"Account status is '{status}'. Please contact Administrator.", None

    # Update last login timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_dict["id"]))
    conn.commit()
    conn.close()

    # Don't return hashes in user session
    del user_dict["password_hash"]
    del user_dict["salt"]
    return True, "Login successful!", user_dict


def get_pending_users() -> list[dict]:
    """Retrieves all users with status 'pending'."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, username, full_name, role, status, created_at FROM users WHERE status = 'pending' ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_users() -> list[dict]:
    """Retrieves all registered users."""
    init_auth_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, username, full_name, role, status, created_at, approved_by, approved_at, last_login FROM users ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def approve_user(username: str, approved_by: str) -> bool:
    """Approves a pending user request."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE users SET status = 'approved', approved_by = ?, approved_at = ? WHERE username = ?", (approved_by, now, username))
    conn.commit()
    success = c.rowcount > 0
    conn.close()
    return success


def reject_user(username: str) -> bool:
    """Rejects a pending user request."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET status = 'rejected' WHERE username = ?", (username,))
    conn.commit()
    success = c.rowcount > 0
    conn.close()
    return success


def update_user_role(username: str, new_role: str) -> bool:
    """Updates a user's role to 'admin' or 'user'."""
    if new_role not in ["admin", "user"]:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, username))
    conn.commit()
    success = c.rowcount > 0
    conn.close()
    return success


def delete_user(username: str) -> bool:
    """Deletes a user account (cannot delete main admin)."""
    if username == "admin":
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ? AND username != 'admin'", (username,))
    conn.commit()
    success = c.rowcount > 0
    conn.close()
    return success
