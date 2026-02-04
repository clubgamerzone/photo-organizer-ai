from app.config import default_config
from app.db import connect, init_db
from app.db import init_ai_tables

init_ai_tables(conn)

cfg = default_config()
conn = connect(cfg.db_path)
init_db(conn)

print("✅ Database created successfully")

conn.close()