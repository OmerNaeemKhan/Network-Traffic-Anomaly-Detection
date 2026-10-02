import sqlite3
from pathlib import Path
import pandas as pd

# ============================================================
# NETWORK TRAFFIC ANOMALY DETECTION - SQL ANALYSIS QUERIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "Outputs" / "network_traffic.db"

print("=" * 70)
print("NETWORK TRAFFIC SQL ANALYSIS")
print("=" * 70)

if not DB_PATH.exists():
    print("\nERROR: Database not found.")
    print(f"Expected location: {DB_PATH}")
    raise SystemExit(1)

connection = sqlite3.connect(DB_PATH)

try:
    cursor = connection.cursor()

    # Get table columns
    cursor.execute("PRAGMA table_info(network_traffic)")
    columns = [row[1] for row in cursor.fetchall()]

    print("\n[OK] Connected to SQL database")
    print("[OK] Table: network_traffic")

    # QUERY 1: Total records
    print("\n" + "-" * 70)
    print("QUERY 1: TOTAL NETWORK TRAFFIC RECORDS")
    print("-" * 70)

    total = pd.read_sql_query(
        "SELECT COUNT(*) AS total_records FROM network_traffic",
        connection
    )
    print(total.to_string(index=False))

    # QUERY 2: Display sample records
    print("\n" + "-" * 70)
    print("QUERY 2: SAMPLE NETWORK TRAFFIC RECORDS")
    print("-" * 70)

    sample = pd.read_sql_query(
        "SELECT * FROM network_traffic LIMIT 10",
        connection
    )
    print(sample.to_string(index=False))

    # Find the prediction/anomaly classification column
    prediction_column = None

    candidates = [
        "prediction",
        "Prediction",
        "predicted_label",
        "Predicted_Label",
        "anomaly",
        "Anomaly",
        "label",
        "Label"
    ]

    for candidate in candidates:
        if candidate in columns:
            prediction_column = candidate
            break

    # QUERY 3: Classification distribution
    if prediction_column:
        print("\n" + "-" * 70)
        print(f"QUERY 3: TRAFFIC CLASSIFICATION DISTRIBUTION")
        print("-" * 70)

        distribution_query = f'''
            SELECT
                "{prediction_column}" AS classification,
                COUNT(*) AS record_count,
                ROUND(
                    COUNT(*) * 100.0 /
                    (SELECT COUNT(*) FROM network_traffic),
                    2
                ) AS percentage
            FROM network_traffic
            GROUP BY "{prediction_column}"
            ORDER BY record_count DESC
        '''

        distribution = pd.read_sql_query(
            distribution_query,
            connection
        )
        print(distribution.to_string(index=False))

        # QUERY 4: Largest groups using SQL
        print("\n" + "-" * 70)
        print("QUERY 4: TOP TRAFFIC GROUPS")
        print("-" * 70)

        top_groups_query = f'''
            SELECT
                "{prediction_column}" AS classification,
                COUNT(*) AS total
            FROM network_traffic
            GROUP BY "{prediction_column}"
            ORDER BY total DESC
        '''

        top_groups = pd.read_sql_query(
            top_groups_query,
            connection
        )
        print(top_groups.to_string(index=False))

    else:
        print("\n[INFO] No prediction column automatically identified.")
        print("Available columns:")
        for column in columns:
            print(f" - {column}")

    # QUERY 5: Database information
    print("\n" + "-" * 70)
    print("QUERY 5: DATABASE SUMMARY")
    print("-" * 70)

    summary = pd.DataFrame({
        "Metric": [
            "Database",
            "Table",
            "Total Columns",
            "Total Records"
        ],
        "Value": [
            DB_PATH.name,
            "network_traffic",
            len(columns),
            pd.read_sql_query(
                "SELECT COUNT(*) AS count FROM network_traffic",
                connection
            ).iloc[0]["count"]
        ]
    })

    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("SQL ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)

finally:
    connection.close()
    print("\nSQL database connection closed.")