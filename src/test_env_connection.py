from sqlalchemy import text

from db_connection import engine

with engine.connect() as connection:
    result = connection.execute(
        text("SELECT current_database(), current_user;")
    )

    print(result.fetchone())