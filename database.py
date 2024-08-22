import sqlite3

def create_database():
    conn = sqlite3.connect('egg_detection.db')
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        is_member INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    ''')

    # Create payments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        amount REAL NOT NULL,
        payment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        payment_method TEXT,
        payment_status TEXT,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    );
    ''')

    # Create daily_limits table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_limits (
        limit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        date DATE,
        image_count INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    );
    ''')

    # Create image_submissions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_submissions (
        submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        image_path TEXT,
        is_fertilized INTEGER,  -- 0 for not fertilized, 1 for fertilized
        submission_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    );
    ''')

    # Create indexes for frequently queried fields
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_limits_user_date ON daily_limits (user_id, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_submissions_user ON image_submissions (user_id)')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database and tables created successfully.")
