import sqlite3
from pathlib import Path
import pandas as pd

# ============================================================
# NETWORK TRAFFIC ANOMALY DETECTION - SQL DATABASE SETUP
# ============================================================

# Project folder (where this setup_database.py file is located)
BASE_DIR = Path(__file__).resolve().parent

# File locations
CSV_PATH = BASE_DIR / "Outputs" / "prediction_results.csv"
DB_PATH = BASE_DIR / "Outputs" / "network_traffic.db"

print("=" * 65)
print("NETWORK TRAFFIC SQL DATABASE SETUP")
print("=" * 65)

# ------------------------------------------------------------
# 1. CHECK CSV FILE
# ------------------------------------------------------------
if not CSV_PATH.exists():
    print(f"\nERROR: CSV file not found:")
    print(CSV_PATH)
    print("\nExpected location:")
    print("Outputs/prediction_results.csv")
    raise SystemExit(1)

print(f"\n[OK] Found CSV file:")
print(CSV_PATH)

# ------------------------------------------------------------
# 2. LOAD CSV SAFELY
# ------------------------------------------------------------
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"\nERROR reading CSV: {e}")
    raise SystemExit(1)

print(f"\n[OK] CSV loaded successfully")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns found:")
for column in df.columns:
    print(f" - {column}")

# ------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ------------------------------------------------------------
df.columns = [
    str(column).strip()
    .replace(" ", "_")
    .replace("-", "_")
    .replace("/", "_")
    .replace("(", "")
    .replace(")", "")
    for column in df.columns
]

# ------------------------------------------------------------
# 4. CREATE SQLITE DATABASE
# ------------------------------------------------------------
print(f"\nCreating SQLite database:")
print(DB_PATH)

connection = sqlite3.connect(DB_PATH)

try:
    # --------------------------------------------------------
    # 5. IMPORT DATA INTO SQL TABLE
    # --------------------------------------------------------
    print("\nImporting data into SQL table: network_traffic ...")

    df.to_sql(
        name="network_traffic",
        con=connection,
        if_exists="replace",
        index=False
    )

    print("[OK] Data imported successfully")

    # --------------------------------------------------------
    # 6. CREATE INDEXES WHERE POSSIBLE
    # --------------------------------------------------------
    cursor = connection.cursor()

    # Get actual SQL column names
    cursor.execute("PRAGMA table_info(network_traffic)")
    table_columns = [row[1] for row in cursor.fetchall()]

    # Try to create indexes on useful columns if they exist
    possible_index_columns = [
        "prediction",
        "Predicted_Label",
        "predicted_label",
        "anomaly",
        "Anomaly",
        "label",
        "Label"
    ]

    for column in possible_index_columns:
        if column in table_columns:
            safe_index_name = f"idx_{column.lower()}"
            try:
                cursor.execute(
                    f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" '
                    f'ON network_traffic ("{column}")'
                )
                print(f"[OK] Index created for: {column}")
            except sqlite3.Error:
                pass

    connection.commit()

    # --------------------------------------------------------
    # 7. VERIFY TOTAL RECORDS
    # --------------------------------------------------------
    cursor.execute("SELECT COUNT(*) FROM network_traffic")
    total_records = cursor.fetchone()[0]

    print("\n" + "=" * 65)
    print("SQL DATABASE VERIFICATION")
    print("=" * 65)

    print(f"\nTotal records in SQL: {total_records:,}")

    # --------------------------------------------------------
    # 8. SHOW TABLE STRUCTURE
    # --------------------------------------------------------
    print("\nSQL Table Columns:")
    for column in table_columns:
        print(f" - {column}")

    # --------------------------------------------------------
    # 9. SHOW SAMPLE DATA
    # --------------------------------------------------------
    print("\nFirst 5 SQL Records:")
    sample = pd.read_sql_query(
        "SELECT * FROM network_traffic LIMIT 5",
        connection
    )

    print(sample.to_string(index=False))

    # --------------------------------------------------------
    # 10. CHECK POSSIBLE PREDICTION / ANOMALY COLUMN
    # --------------------------------------------------------
    prediction_column = None

    priority_names = [
        "prediction",
        "Prediction",
        "predicted_label",
        "Predicted_Label",
        "anomaly",
        "Anomaly",
        "label",
        "Label"
    ]

    for column in priority_names:
        if column in table_columns:
            prediction_column = column
            break

    if prediction_column:
        print("\n" + "-" * 65)
        print(f"SQL DISTRIBUTION BY: {prediction_column}")
        print("-" * 65)

        query = f'''
            SELECT
                "{prediction_column}" AS classification,
                COUNT(*) AS total_records
            FROM network_traffic
            GROUP BY "{prediction_column}"
            ORDER BY total_records DESC
        '''

        distribution = pd.read_sql_query(query, connection)
        print(distribution.to_string(index=False))

    else:
        print("\n[INFO] No standard prediction column was automatically identified.")
        print("The database was still created successfully.")
        print("We can use the actual column names above for custom SQL queries.")

    # --------------------------------------------------------
    # 11. FINAL SUCCESS MESSAGE
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUCCESS!")
    print("=" * 65)
    print(f"\nDatabase created successfully:")
    print(DB_PATH)
    print(f"\nTable created: network_traffic")
    print(f"Records imported: {total_records:,}")
    print("\nYour project now includes:")
    print("  [OK] Python")
    print("  [OK] Machine Learning / AI")
    print("  [OK] Streamlit Dashboard")
    print("  [OK] SQL Database")
    print("\nYou can now demonstrate real SQL queries on your network data.")

finally:
    connection.close()
    print("\nDatabase connection closed.")