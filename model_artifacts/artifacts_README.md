# ⚙️ MLOps Guide: Generating Model Artifacts

The final Machine Learning model and its vector embeddings are not committed to this repository due to file size constraints. This is standard MLOps practice, where code is separated from compiled/trained assets and managed via deployment jobs.

### Prerequisites

You must have already run `docker compose up -d` to ensure the `postgres_db` service is running and contains the full transaction data.

### Step 1: Run the Training Job

Use the `docker compose run` command to execute the training script. This job will connect to the database, pull all the historical data, train the Hybrid Model, and save the final files to the mounted volume.

```bash
docker compose run --rm training_service python train_model.py
```

### Expected Output (Example Logs)

The script will take a significant amount of time (45 minutes to 2 hours) to complete the training epochs. The final successful output in your console will look similar to this:

```plaintext
1. Acquiring high-volume data from PostgreSQL...
   -> Final Loaded Training Transactions DF size: 35,000,000
2. Building interaction and feature matrices...
...
3. Training Hybrid Model and generating vectors...
   -> Item Embeddings Shape: (105542, 64)
4. Saving model artifacts to /app/model_artifacts...
   -> Artifacts saved: Model, Embeddings, and ID Mappings.

ML Model Training Pipeline completed successfully.
```

### Step 2: Verify Artifacts

After the script completes, the following essential files will be present in your local `./model_artifacts` directory, making the API ready for deployment:

1.  `hybrid_model.joblib` (The trained model object)
2.  `item_embeddings.npy` (The vector search map)
3.  `user_embeddings.npy` (The user profile vectors)
4.  `mapping_dataset.joblib` (The ID-to-Index decoder ring)