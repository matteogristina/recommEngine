import os
import sys
import pandas as pd
import re
import psycopg2

def clean_prod_name(prod_name: str) -> str:
    """
    Cleans a product name string by:
    1. Converting to lowercase.
    2. Removing package count references (e.g., '2p', '4p', '2-pack').
    3. Removing specific phrases like 'l/s' and 's/s'.
    4. Removing other special characters while preserving 'w/'.
    5. Normalizing whitespace.
    6. Stripping leading/trailing whitespace.
    """
    if not isinstance(prod_name, str):
        return ""

    # Step 1: Convert to lowercase
    text = prod_name.lower()

    # Step 2: Remove specific phrases and abbreviations
    # Use re.sub with a list of patterns to remove.
    # The \b is a word boundary to prevent partial matches (e.g., in a word like "slouch").
    patterns_to_remove = [
        r'\b\d+p\b',  # e.g., "2p", "5p"
        r'\d+-pack',  # e.g., "2-pack"
        r'\bl/s\b',  # "l/s" for long sleeve
        r'\bs/s\b',  # "s/s" for short sleeve
        r'\b(\d)\b',  # "(x)" for pkg count
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)

    # Step 3: Remove unwanted special characters, BUT keep 'w/'
    # The regex below will remove any character that is NOT a lowercase letter,
    # a digit, a space, or a forward slash. This is more precise.
    # Note: We must also handle the parentheses and other specific symbols.
    text = re.sub(r'[()^,]', '', text)

    # Step 4: Normalize multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)

    # Step 5: Trim leading and trailing whitespace
    text = text.strip()

    return text


def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using environment variables.
    Returns: A database connection object.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres_db"),
            database=os.getenv("DB_NAME", "recommender_db"),
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD", "password")
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        sys.exit(1)


def ingest_articles(articles_df: pd.DataFrame, cur, conn):
    """
    Ingests data from the articles dataframe into the 'products' and 'articles' tables.
    """
    print("Starting ingestion of articles data...")

    # Pass 1: Ingest unique values into lookup tables
    print("  - Ingesting lookup tables...")

    def insert_unique_values(table_name, id_col, name_col):
        unique_data = set(zip(articles_df[id_col], articles_df[name_col]))
        query = f"INSERT INTO {table_name} ({id_col}, {name_col}) VALUES (%s, %s) ON CONFLICT ({id_col}) DO NOTHING;"
        cur.executemany(query, list(unique_data))

    insert_unique_values("product_types", "product_type_no", "product_type_name")
    insert_unique_values("apparel_collections", "index_group_no", "index_group_name")
    insert_unique_values("sections", "section_no", "section_name")
    insert_unique_values("garment_groups", "garment_group_no", "garment_group_name")
    insert_unique_values("perceived_colors", "perceived_colour_master_id", "perceived_colour_master_name")

    # Pass 2: Ingest into main tables
    print("  - Ingesting products and articles...")
    products_to_insert = []
    articles_to_insert = []

    for _, row in articles_df.iterrows():
        # Clean the product name on the fly
        cleaned_prod_name = clean_prod_name(row['prod_name'])

        # Prepare data for 'products' table
        products_data = (row['product_code'], cleaned_prod_name, row['detail_desc'],
                         row['product_type_no'], row['index_group_no'],
                         row['section_no'], row['garment_group_no'])
        products_to_insert.append(products_data)

        # Prepare data for 'articles' table (using relative image path)
        image_path = f"/images/{row['article_id'][0:3]}/{row['article_id']}.jpg"
        articles_data = (row['article_id'], image_path, row['product_code'],
                         row['perceived_colour_master_id'])
        articles_to_insert.append(articles_data)

    products_query = """
                     INSERT INTO products (product_code, prod_name, detail_desc, product_type_no, \
                                           index_group_no, section_no, garment_group_no) \
                     VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (product_code) DO NOTHING; \
                     """
    cur.executemany(products_query, products_to_insert)

    articles_query = """
                     INSERT INTO articles (article_id, image_path, product_code, perceived_colour_master_id) \
                     VALUES (%s, %s, %s, %s) ON CONFLICT (article_id) DO NOTHING; \
                     """
    cur.executemany(articles_query, articles_to_insert)

    conn.commit()
    print("Articles ingestion complete.")


def ingest_customers(df: pd.DataFrame, cur, conn):
    """Ingests data from the customers.csv portion of the dataframe."""
    print("Starting ingestion of customers data...")

    # Create a list to hold the data tuples for bulk insertion
    customers_to_insert = []

    # Iterate over the DataFrame to prepare data for insertion
    # The columns are: customer_id, club_member_status, fashion_news_freq, age, postal_code
    for _, row in df.iterrows():
        # Corrected code to handle NaN values in the 'age' column
        age_value = int(row['age']) if pd.notna(row['age']) else None

        customer_data = (
            row['customer_id'],
            row['club_member_status'],
            row['fashion_news_frequency'],
            age_value,
            row['postal_code']
        )
        customers_to_insert.append(customer_data)

    # Define the SQL query for bulk insertion
    # We use ON CONFLICT DO NOTHING to make the script idempotent,
    # preventing errors if a customer ID already exists from a previous run.
    query = """
            INSERT INTO customers (customer_id, club_member_status, fashion_news_freq, age, postal_code) \
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (customer_id) DO NOTHING; \
            """

    # Use psycopg2's executemany for efficient bulk insertion
    cur.executemany(query, customers_to_insert)

    # Print the number of rows affected by the last command
    print(f"  - Ingested {cur.rowcount} unique customers.")

    conn.commit()
    print("Customers ingestion complete.")


# Function to ingest data from the transactions.csv file
def ingest_transactions(csv_file_path: str, cur, conn):
    """
    Ingests transaction data from a CSV file in chunks to handle large files.
    """
    print("Starting chunked ingestion of transactions data...")

    chunksize = 1_000_000
    total_chunks = 0
    total_rows_inserted = 0

    try:
        # Use pandas to read the CSV in chunks
        for chunk_df in pd.read_csv(
                csv_file_path,
                chunksize=chunksize,
                dtype={'article_id': str, 'customer_id': str},  # Ensure correct data types
                engine='python'  # Use the robust python engine
        ):
            total_chunks += 1
            print(f"  - Processing chunk #{total_chunks}...")

            transactions_to_insert = []

            # Inner loop to prepare data from the current chunk
            for index, row in chunk_df.iterrows():
                t_dat = pd.to_datetime(row['t_dat']).date()

                transaction_data = (
                    t_dat,
                    row['customer_id'],
                    row['article_id'],
                    row['price']
                )
                transactions_to_insert.append(transaction_data)

            # Your SQL query for insertion
            transactions_query = """
                                 INSERT INTO transactions (t_dat, customer_id, article_id, price)
                                 VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING; \
                                 """

            # Bulk insert the current chunk
            cur.executemany(transactions_query, transactions_to_insert)

            # Commit the transaction for this chunk
            total_rows_inserted += cur.rowcount
            conn.commit()
            print(f"  - Chunk #{total_chunks} ingested. Total rows: {total_rows_inserted}")

    except Exception as e:
        print(f"An error occurred during chunked ingestion: {e}")
        conn.rollback()
        raise  # Re-raise the exception to stop the main process

    print("Transactions ingestion complete.")


def main():
    """
    Main function to orchestrate the entire ingestion process.
    """
    conn = None
    cur = None
    try:
        print("Getting connection...")
        conn = get_db_connection()
        cur = conn.cursor()

        print("Read articles...")

        # Load articles and customers as before (they are small enough)
        articles_df = pd.read_csv('/app/data/articles.csv', dtype={'article_id': str}, engine='python')

        # Step 1: Ingest articles data.
        ingest_articles(articles_df, cur, conn)

        print("Read customers...")

        customers_df = pd.read_csv('/app/data/customers.csv', engine='python')

        # Step 2: Ingest customer data.
        ingest_customers(customers_df, cur, conn)

        # Step 3: Ingest the transactions data using the new chunking function.
        # Pass the file path directly instead of the DataFrame.
        ingest_transactions('/app/data/transactions_train.csv', cur, conn)

        print("All data ingestion completed successfully.")

    except (Exception, psycopg2.DatabaseError) as e:
        print(f"An error occurred during ingestion: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    main()