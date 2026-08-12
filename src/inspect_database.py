from sqlalchemy import create_engine, text

DB_USER = "postgres"
DB_PASSWORD = "test"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "credit_risk"

connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_string)

with engine.connect() as connection:

    result = connection.execute(
        text("""
            SELECT
                current_database(),
                current_user,
                inet_server_addr(),
                inet_server_port();
        """)
    )

    print(result.fetchone())

    for table in [
        "accounts",
        "customers",
        "delinquency",
        "economic_data",
        "payments"
    ]:
        result = connection.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        )
        print(f"{table}: {result.scalar()} rows")