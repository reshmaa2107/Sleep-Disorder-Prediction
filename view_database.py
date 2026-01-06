import sqlite3
import pandas as pd

# 1. Connect to the database
connection = sqlite3.connect('sleep_data.db')

# 2. Query all data
try:
    # We use pandas to make it look like a nice Excel table
    df = pd.read_sql_query("SELECT * FROM users", connection)
    
    # 3. Print the data
    if df.empty:
        print("📭 The database is empty. Go to the website and make a prediction first!")
    else:
        print("📊 HERE IS YOUR USER DATA:")
        print("------------------------------------------------")
        print(df)
        print("------------------------------------------------")

except Exception as e:
    print(f"❌ Error: {e}")

# 4. Close connection
connection.close()