import os
import sys
import pandas as pd
import numpy as np
import psycopg2
import joblib
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.data import Dataset  # Important for LightFM feature management
from typing import Tuple, Dict, Any

# --- CONFIGURATION ---
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR")
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "hybrid_model.joblib")
ITEM_EMBEDDINGS_FILE = os.path.join(ARTIFACTS_DIR, "item_embeddings.npy")
USER_EMBEDDINGS_FILE = os.path.join(ARTIFACTS_DIR, "user_embeddings.npy")
DATASET_FILE = os.path.join(ARTIFACTS_DIR, "mapping_dataset.joblib")
N_COMPONENTS = 64  # Dimension of the embedding vectors

TRAINING_CUTOFF_DATE = '2020-09-17'


def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using environment variables.
    """
    try:
        # DB_HOST is 'postgres_db' in the Docker network
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        sys.exit(1)


# Step 1: Execute SQL queries and return raw dataframes
def acquire_data(conn) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Acquires transactions, article metadata, and customer metadata from PostgreSQL.

    The transactions data is streamed in chunks for memory efficiency and
    is loaded with a temporal split for training.

    Returns: (transactions_df, articles_df, customers_df)
    """
    print("1. Acquiring high-volume data from PostgreSQL...")

    # --- 1.1 Articles Data (Item Features - Small enough to load entirely) ---
    # Joins 'articles' with 'products' to get content features.
    articles_query = """
                     SELECT t1.article_id,
                            t2.product_type_no,
                            t2.garment_group_no,
                            t1.perceived_colour_master_id
                     FROM articles t1
                              JOIN products t2 ON t1.product_code = t2.product_code;
                     """
    articles_df = pd.read_sql_query(articles_query, conn)
    print(f"   -> Loaded Articles (metadata) DF size: {len(articles_df):,}")

    # --- 1.2 Customers Data (User Features - Small enough to load entirely) ---
    # Extracts demographic data for the User Feature Matrix.
    customers_query = """
                      SELECT customer_id, age, club_member_status
                      FROM customers;
                      """
    customers_df = pd.read_sql_query(customers_query, conn)
    print(f"   -> Loaded Customers (metadata) DF size: {len(customers_df):,}")

    # --- 1.3 Transactions Data (SCALABLE FIX) ---
    transactions_query = f"""
                             SELECT customer_id, article_id 
                             FROM transactions
                             WHERE t_dat < '{TRAINING_CUTOFF_DATE}'
                             ORDER BY customer_id, t_dat ASC;
                             """

    # Configuration for streaming
    cursor_name = "transactions_stream"
    chunk_size = 500_000

    total_rows = 0
    all_transactions_dfs = []  # <<< CHANGE: This will hold DataFrames (not tuples)

    conn.commit()

    with conn.cursor(name=cursor_name) as cur:
        cur.execute(transactions_query)

        while True:
            rows = cur.fetchmany(chunk_size)

            if not rows:
                break

            # --- CRITICAL CHANGE: Create a DataFrame from the chunk immediately ---
            chunk_df = pd.DataFrame(rows, columns=['customer_id', 'article_id'])
            all_transactions_dfs.append(chunk_df)

            total_rows += len(rows)

            # --- PROGRESS CHECK PRINT ---
            if total_rows % 1_000_000 == 0:
                print(f"   -> Progress: Fetched {total_rows:,} training transactions...")

    # --- FINAL STEP: CONCATENATE ALL THE SMALL DATAFRAMES INTO ONE ---
    print(f"   -> Concatenating {len(all_transactions_dfs)} DataFrames...")
    transactions_df = pd.concat(all_transactions_dfs, ignore_index=True)

    print(f"   -> Final Loaded Training Transactions DF size: {total_rows:,}")

    # Return all three DataFrames
    return transactions_df, articles_df, customers_df


# Step 2: Create interaction and feature matrices
def build_matrices(data_frames: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
    """Creates sparse interaction, item feature, and user feature matrices for LightFM."""
    print("2. Building interaction and feature matrices...")

    transactions_df, articles_df, customers_df = data_frames

    # --- Step 2.1: Define the UNIVERSES ---
    user_universe = transactions_df['customer_id'].unique()
    item_universe = articles_df['article_id'].unique()

    # --- Step 2.2: CRITICAL FIX - Filter Customers to Match the Training Set ---
    # Only keep customers who actually appear in the training transaction data.
    customers_df_filtered = customers_df[
        customers_df['customer_id'].isin(user_universe)
    ].copy()

    print(f"   -> Reduced Customer Features from {len(customers_df):,} to {len(customers_df_filtered):,}")

    # 2.3 Fit the Dataset object with all unique features and IDs
    dataset = Dataset()
    dataset.fit(
        # The universe of users is defined by those in the training transactions
        users=user_universe,
        items=item_universe,

        # Fit all available feature types
        item_features=[str(c) for c in articles_df['garment_group_no'].unique()],
        user_features=customers_df_filtered['club_member_status'].unique()  # Use filtered DF for feature fitting
    )

    # 2.4 Interaction Matrix (The Core Input)
    (interactions, weights) = dataset.build_interactions(
        transactions_df.apply(lambda x: (x['customer_id'], x['article_id']), axis=1).values
    )

    # 2.5 Item Feature Matrix
    item_features = dataset.build_item_features(
        articles_df.apply(lambda x: (x['article_id'], [str(x['garment_group_no'])]), axis=1).values
    )

    # --- Step 2.6: BUILD USER FEATURES (Using the filtered DataFrame) ---
    user_features = dataset.build_user_features(
        customers_df_filtered.apply(lambda x: (x['customer_id'], [x['club_member_status']]), axis=1).values
    )

    print(f"   -> Interactions Matrix Shape: {interactions.shape}")
    print(f"   -> User Features Matrix Shape: {user_features.shape}")

    return interactions, item_features, user_features, dataset


# Step 3: Train the model and generate embedding vectors
def train_and_embed(matrices: Tuple[coo_matrix, coo_matrix, coo_matrix, Dataset]):
    """Trains the Hybrid LightFM model and generates embeddings."""
    print("3. Training Hybrid Model and generating vectors...")

    interactions, item_features, user_features, dataset = matrices

    # Instantiate LightFM - loss='warp' is good for implicit feedback
    model = LightFM(loss='warp', no_components=N_COMPONENTS)

    # Train the model using both interactions and item features (Hybrid!)
    model.fit(
        interactions=interactions,
        item_features=item_features,
        epochs=20,  # Reduced epochs for demonstration, use more in production
        num_threads=4
    )

    # Generate Embeddings (The core intelligence)
    # Note: Embeddings are generated after training
    item_embeddings = model.get_item_representations(features=item_features)[0]
    user_embeddings = model.get_user_representations()[0]

    print(f"   -> Item Embeddings Shape: {item_embeddings.shape}")
    print(f"   -> User Embeddings Shape: {user_embeddings.shape}")

    return model, item_embeddings, user_embeddings, dataset


# Step 4: Save files to the mounted volume
def save_artifacts(model, item_embeddings, user_embeddings, dataset):
    """Saves the trained model, embedding matrices, and ID mappings (MLOps Artifacts)."""
    print(f"4. Saving model artifacts to {ARTIFACTS_DIR}...")

    # Ensure the artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save the model object
    joblib.dump(model, MODEL_FILE)

    # --- CRITICAL ADDITION: Save the Dataset object which holds all mappings ---
    joblib.dump(dataset, DATASET_FILE)

    # Save the embedding matrices (NumPy format)
    np.save(ITEM_EMBEDDINGS_FILE, item_embeddings)
    np.save(USER_EMBEDDINGS_FILE, user_embeddings)

    print("   -> Artifacts saved: Model, Embeddings, and ID Mappings.")


if __name__ == "__main__":
    conn = None
    try:
        conn = get_db_connection()

        # 1. Acquire Data
        raw_data = acquire_data(conn)

        # 2. Build Matrices (The Input Map)
        # Note the updated unpacking
        matrices = build_matrices(raw_data)

        # 3. Train Model and Generate Vectors (The Intelligence)
        # Note the updated unpacking
        model, item_embeddings, user_embeddings, dataset = train_and_embed(matrices)

        # 4. Save Artifacts (The Deliverable)
        # Pass the dataset object
        save_artifacts(model, item_embeddings, user_embeddings, dataset)

        print("\nML Model Training Pipeline completed successfully.")

    except Exception as e:
        print(f"\nFATAL ERROR during ML Pipeline: {e}")
        sys.exit(1)

    finally:
        if conn:
            conn.close()