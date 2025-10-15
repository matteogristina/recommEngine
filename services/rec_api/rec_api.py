from fastapi import FastAPI, HTTPException
import os, joblib, numpy as np, redis, sys
from typing import List, Dict

# --- GLOBAL ARTIFACTS AND CLIENTS (Loaded at Startup) ---
# Define paths based on the volume mount in docker-compose.yml
ARTIFACTS_DIR = "/app/model_artifacts"
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "hybrid_model.joblib")
EMBEDDINGS_FILE = os.path.join(ARTIFACTS_DIR, "item_embeddings.npy")  # Note: Assuming this also holds user embeds
MAPPING_FILE = os.path.join(ARTIFACTS_DIR, "mapping_dataset.joblib")

MODEL = None
ITEM_EMBEDDINGS = None
USER_EMBEDDINGS = None  # We will load this separately for clarity
REDIS_CLIENT = None

# CRITICAL MAPPINGS (Loaded from joblib file)
USER_ID_TO_INDEX: Dict[str, int] = {}
ITEM_ID_TO_INDEX: Dict[str, int] = {}
ITEM_INDEX_TO_ID: Dict[int, str] = {}  # This one was missing!

app = FastAPI(title="Recommendation API Microservice")


@app.on_event("startup")
def load_resources():
    global MODEL, ITEM_EMBEDDINGS, USER_EMBEDDINGS, REDIS_CLIENT
    global USER_ID_TO_INDEX, ITEM_ID_TO_INDEX, ITEM_INDEX_TO_ID

    print("--- 1. LOADING ARTIFACTS ---")
    try:
        # Load the LightFM Dataset object which contains all mappings
        dataset = joblib.load(MAPPING_FILE)

        # 1.1 Load Model and Embeddings (Intelligence)
        MODEL = joblib.load(MODEL_FILE)

        # NOTE: If user_embeddings and item_embeddings are saved separately, adjust loading here.
        # For robustness, we will assume ITEM_EMBEDDINGS holds the item vectors.
        # The training script should be saving user embeddings separately.
        # We will assume separate saving as per the training script outline.
        ITEM_EMBEDDINGS = np.load(os.path.join(ARTIFACTS_DIR, "item_embeddings.npy"))
        USER_EMBEDDINGS = np.load(os.path.join(ARTIFACTS_DIR, "user_embeddings.npy"))

        # 1.2 Extract Mappings (The 'Decoder Ring')
        # LightFM's Dataset provides these directly.
        USER_ID_TO_INDEX = dataset.mapping()[0]  # user_id_mapping
        ITEM_ID_TO_INDEX = dataset.mapping()[2]  # item_id_mapping

        # Invert the item map for the final output conversion
        ITEM_INDEX_TO_ID = {v: k for k, v in ITEM_ID_TO_INDEX.items()}

        # 1.3 Initialize Redis client and check connection
        REDIS_CLIENT = redis.Redis(host='redis_cache', port=6379, decode_responses=True)
        REDIS_CLIENT.ping()

        print("--- 2. RESOURCES LOADED AND REDIS CONNECTED ---")
        print(f"Loaded {len(USER_ID_TO_INDEX):,} user mappings and {len(ITEM_ID_TO_INDEX):,} item mappings.")

    except Exception as e:
        print(f"FATAL ERROR: Could not load model artifacts or connect to Redis: {e}")
        # In a production setting, you would want the container to fail to start
        sys.exit(1)


# --- FUNCTION 1: Stage 1 Candidate Generation (FIXED) ---
def get_recommendations_and_scores(user_index: int, k: int = 100) -> np.ndarray:
    """
    Stage 1: Generates the top K candidate items and their scores.
    """
    global MODEL, ITEM_EMBEDDINGS

    # 1. Prediction Setup
    all_item_indices = np.arange(ITEM_EMBEDDINGS.shape[0])

    # --- CRITICAL FIX: Broadcast the single user_index to the length of all items ---
    # This creates a NumPy array of size 105,542, where every element is the user_index.
    broadcasted_user_indices = np.full_like(all_item_indices, user_index)

    # Model now predicts scores for 105,542 user-item pairs, where the user is the same.
    scores = MODEL.predict(user_ids=broadcasted_user_indices, item_ids=all_item_indices)

    # 2. Ranking and Selection
    # Get the indices of the top K scores
    top_k_indices = np.argsort(-scores)[:k]

    # Return a 2D array of [item_index, score] for the top K candidates
    # We must ensure scores[top_k_indices] is cast to float32 before stacking for consistency
    candidates = np.stack((top_k_indices.astype(np.float32), scores[top_k_indices].astype(np.float32)), axis=1)

    return candidates  # [Item_Index, Score]


# --- FUNCTION 2: Stages 2 & 3 Stochastic Search (MAPPING INTEGRATED) ---
def stochastic_search_ranking(user_id: str, candidates: np.ndarray, num_return: int = 10) -> List[str]:
    """
    Stages 2 & 3: Applies real-time re-ranking and stochastic selection.
    """
    global REDIS_CLIENT, ITEM_ID_TO_INDEX, ITEM_INDEX_TO_ID

    # --- Stage 2: Real-Time Re-Ranking Heuristics ---

    # 1. Get Live User Intent from Redis
    recent_items_ids = REDIS_CLIENT.lrange(f"user_history:{user_id}", 0, 9)

    # Convert recent item IDs (strings) to indices for comparison
    recent_items_indices = [ITEM_ID_TO_INDEX.get(i) for i in recent_items_ids if i in ITEM_ID_TO_INDEX]

    # Filter out any candidates that are in the user's recent history
    # CRITICAL: We must convert the candidates NumPy array back to a list/tuple to filter easily
    candidates_list = [(int(idx), score) for idx, score in candidates]

    filtered_candidates = [
        (idx, score) for idx, score in candidates_list
        if idx not in recent_items_indices
    ]

    # --- Stage 3: Stochastic Selection (Exploitation vs. Exploration) ---

    if len(filtered_candidates) <= num_return:
        return [str(ITEM_INDEX_TO_ID.get(int(idx))) for idx, score in filtered_candidates]


    # Sort by score again (after filtering)
    filtered_candidates.sort(key=lambda x: x[1], reverse=True)

    # 1. Define Buckets
    num_exploit = int(num_return * 0.8)
    num_explore = num_return - num_exploit

    # 2. The Two Pools
    # We use a safety buffer to ensure we have enough items
    exploit_pool = filtered_candidates[:num_exploit * 2]
    explore_pool = filtered_candidates[num_exploit * 2:]

    # 3. Stochastic Selection
    final_selection = []

    # A. Exploit (Deterministic Selection): Select the top N from the exploit pool
    final_selection.extend([idx for idx, score in exploit_pool[:num_exploit]])

    # B. Explore (Random Selection): Select M items randomly from the larger explore pool
    if len(explore_pool) > 0:
        # Get random sample of indices from the exploration pool
        explore_indices = np.random.choice(len(explore_pool), size=min(num_explore, len(explore_pool)), replace=False)

        for i in explore_indices:
            final_selection.append(explore_pool[i][0])  # Append the item index

    # 4. Map and Shuffle

    # Map item indices back to original article IDs using the loaded global map
    final_recommendations = [str(ITEM_INDEX_TO_ID.get(idx)) for idx in final_selection]

    # Final shuffle for presentation (optional, but good for discovery)
    np.random.shuffle(final_recommendations)

    return final_recommendations


# --- The Main API Endpoint (Where it all comes together) ---
@app.get("/api/v1/predict", response_model=List[str])
def predict_recommendations(user_id: str, mode: str = "CROSS_SELL") -> List[str]:
    """
    The Generalized Prediction Endpoint.
    """
    global USER_ID_TO_INDEX

    # 1. Input Validation
    # Use .get() for safety, handle the case where the user has no index
    user_index = USER_ID_TO_INDEX.get(user_id)

    if user_index is None:
        # A new user (cold start). We can raise an error or return default/popular items.
        # For simplicity, we raise an error. In a live system, you'd return the most popular items.
        raise HTTPException(status_code=404, detail="User ID not found in training data.")

    # 2. Candidate Generation (Stage 1)
    candidates_with_scores = get_recommendations_and_scores(user_index, k=100)

    # 3. Stochastic Re-Ranking (Stages 2 & 3)
    final_recommendations = stochastic_search_ranking(user_id, candidates_with_scores, num_return=10)

    return final_recommendations