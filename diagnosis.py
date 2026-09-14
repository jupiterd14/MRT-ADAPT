# test_pg.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

with engine.connect() as conn:
    for t in ["user", "report", "broadcast", "activity_log", "saved_route", "station_data", "activity"]:
        n = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
        print(f"{t}: {n}")