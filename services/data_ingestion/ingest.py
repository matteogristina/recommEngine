import os
import sys
import csv
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor


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


def ingest_data_from_csv(csv_file_path):
    """
    Reads a CSV file and ingests the data into the PostgreSQL database.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Create a list to hold the product data for bulk insertion
        products_to_insert = []

        # Create a list to hold the image data for bulk insertion
        images_to_insert = []

        # Assuming your images are served from a local image_server on port 8000
        # The path 'data' is the name of the directory in your bind mount
        IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://localhost:8000/images")

        # Open the CSV file from the path inside the container
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            print(f"Starting ingestion from {csv_file_path}...")

            for row in reader:
                # 1. Prepare data for the products table
                product_data = (
                    row['sku'],
                    row['name'],
                    row['description'],
                    float(row['price']),
                    row['terms'],
                    row['section'],
                    f"{IMAGE_BASE_URL}/{row['main_image']}.jpg"
                )
                products_to_insert.append(product_data)

                # 2. Prepare data for the product_images table
                # The 'image_downloads' column contains a comma-separated string of filenames
                translation_table = str.maketrans("", "", "['] ")
                image_filenames = row['image_downloads'].translate(translation_table).split(',')
                for filename in image_filenames:
                    if filename.strip():
                        # We will link images after the products have been inserted
                        # and we have a product_id. For now, we'll store the sku.
                        image_url = f"{IMAGE_BASE_URL}/{filename.strip()}.jpg"
                        images_to_insert.append((row['sku'], image_url))

        print("Data prepared. Inserting into database...")

        # Bulk insert into the products table
        # We need to use `RETURNING` to get the `product_id` for the images
        execute_values(
            cur,
            """INSERT INTO products (sku, name, description, price, terms, section, main_image_url) 
               VALUES %s ON CONFLICT (sku) DO NOTHING""",
            products_to_insert
        )

        # Now, we need to handle the images. A more robust way would be to
        # first get all the SKUs and their new product_ids.
        # For simplicity, we assume SKUs are unique and can be used to join.

        # A better approach would involve a temporary table or a two-step process
        # to ensure image insertion is linked correctly.
        # For this example, let's assume we fetch all products and then match up.
        cur.execute("SELECT sku, product_id FROM products")
        sku_to_id_map = {row['sku']: row['product_id'] for row in cur.fetchall()}

        print(sku_to_id_map)

        final_images_to_insert = []
        for sku, image_url in images_to_insert:
            if sku in sku_to_id_map:
                final_images_to_insert.append((sku_to_id_map[sku], image_url))

        # Bulk insert into the product_images table
        execute_values(
            cur,
            "INSERT INTO product_images (product_id, image_url) VALUES %s",
            final_images_to_insert
        )

        conn.commit()
        print(f"Successfully ingested {len(products_to_insert)} products and {len(final_images_to_insert)} images.")


    except (Exception, psycopg2.DatabaseError) as e:
        print(f"An error occurred: {e}")
        #conn.rollback()  # Rollback the transaction on error
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_csv>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    ingest_data_from_csv(csv_file_path)