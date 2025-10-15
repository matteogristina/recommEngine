from fastapi import FastAPI, HTTPException
import os, joblib, numpy as np, redis, sys
from typing import List, Dict, Optional, Any
from scipy.spatial.distance import cosine

# --- GLOBAL ARTIFACTS AND CLIENTS (Loaded at Startup) ---
# ... (Configuration and Artifact Loading remain the same) ...

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR")
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "hybrid_model.joblib")
MAPPING_FILE = os.path.join(ARTIFACTS_DIR, "inference_mappings.joblib")  # Using the clean mappings file

MODEL = None
ITEM_EMBEDDINGS = None
USER_EMBEDDINGS = None
REDIS_CLIENT = None

# CRITICAL MAPPINGS (Loaded from joblib file)
USER_ID_TO_INDEX: Dict[str, int] = {}
ITEM_ID_TO_INDEX: Dict[str, int] = {}
ITEM_INDEX_TO_ID: Dict[int, str] = {}

app = FastAPI(title="Recommendation API Microservice")


@app.on_event("startup")
def load_resources():
    global MODEL, ITEM_EMBEDDINGS, USER_EMBEDDINGS, REDIS_CLIENT
    global USER_ID_TO_INDEX, ITEM_ID_TO_INDEX, ITEM_INDEX_TO_ID

    print("--- 1. LOADING ARTIFACTS ---")
    try:
        # Load Artifacts
        MODEL = joblib.load(MODEL_FILE)
        ITEM_EMBEDDINGS = np.load(os.path.join(ARTIFACTS_DIR, "item_embeddings.npy"))
        USER_EMBEDDINGS = np.load(os.path.join(ARTIFACTS_DIR, "user_embeddings.npy"))

        # Load and extract Mappings from the streamlined joblib file
        mappings = joblib.load(MAPPING_FILE)
        USER_ID_TO_INDEX = mappings['user_id_to_index']
        ITEM_ID_TO_INDEX = mappings['item_id_to_index']
        ITEM_INDEX_TO_ID = {v: k for k, v in ITEM_ID_TO_INDEX.items()}  # Invert for final output

        # Initialize Redis client
        REDIS_CLIENT = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")),
                                   decode_responses=True)
        REDIS_CLIENT.ping()

        print("--- 2. RESOURCES LOADED AND REDIS CONNECTED ---")
        print(f"Loaded {len(USER_ID_TO_INDEX):,} user mappings and {len(ITEM_ID_TO_INDEX):,} item mappings.")

    except Exception as e:
        print(f"FATAL ERROR: Could not load model artifacts or connect to Redis: {e}")
        sys.exit(1)


def safe_item_lookup(item_id_str: str) -> Optional[int]:
    """
    Attempts to find the integer index of an item by checking its string and
    integer representation against the ITEM_ID_TO_INDEX mapping.
    Returns the integer index if found, None otherwise.
    """
    global ITEM_ID_TO_INDEX

    # 1. Try Lookup as the clean STRING (Most likely correct)
    if item_id_str in ITEM_ID_TO_INDEX:
        return ITEM_ID_TO_INDEX[item_id_str]

    # 2. Try Lookup as an INTEGER (The common failure mode)
    try:
        # Convert the string ID to a raw integer
        item_id_int = int(item_id_str)
        if item_id_int in ITEM_ID_TO_INDEX:
            return ITEM_ID_TO_INDEX[item_id_int]
    except ValueError:
        # Not a valid number (e.g., a dummy ID), ignore
        pass

    return None


# --- NEW FUNCTION: The Pure Content-Based Cold-Start Fallback ---
def cold_start_recommendations(target_item_id: str, num_return: int = 10) -> List[str]:
    """
    Solves the cold-start problem by finding items most similar to a single target item
    using vector similarity (Cosine Similarity).
    """
    global ITEM_EMBEDDINGS, ITEM_INDEX_TO_ID

    # --- CRITICAL CHANGE: Use the safe lookup function ---
    target_index = safe_item_lookup(target_item_id)

    # Validation: If the item itself is new/invalid, return empty list.
    if target_index is None:
        print(f"DEBUG: Item ID '{target_item_id}' not found in catalog mapping after string/int check.")
        return []

    # 1. Get the target item's vector
    target_vector = ITEM_EMBEDDINGS[target_index]

    # 2. Calculate Cosine Similarity against ALL item vectors
    # Similarity is calculated via dot product with the whole embedding matrix
    similarities = ITEM_EMBEDDINGS.dot(target_vector)

    # 3. Find the Top N Closest Items
    # We find the top 50 candidates for discovery
    # [1:num_return + 50] skips the item itself (index 0)
    top_indices = np.argsort(similarities)[::-1][1:num_return + 50]

    # 4. Stochastic Selection for Discovery

    # Convert indices to a list
    candidate_indices = top_indices.tolist()

    # Select 10 items randomly from the top 50 candidates (more exploratory for new users)
    final_indices = np.random.choice(candidate_indices, size=num_return, replace=False).tolist()

    # 5. Map and Return
    final_recommendations = [str(ITEM_INDEX_TO_ID.get(idx)) for idx in final_indices]

    # Final shuffle
    np.random.shuffle(final_recommendations)
    return final_recommendations


# --- FUNCTION 1: Stage 1 Candidate Generation (FIXED) ---
def get_recommendations_and_scores(user_index: int, k: int = 100) -> np.ndarray:
    # ... (No functional changes needed here) ...
    global MODEL, ITEM_EMBEDDINGS
    all_item_indices = np.arange(ITEM_EMBEDDINGS.shape[0])
    broadcasted_user_indices = np.full_like(all_item_indices, user_index)
    scores = MODEL.predict(user_ids=broadcasted_user_indices, item_ids=all_item_indices)
    top_k_indices = np.argsort(-scores)[:k]
    candidates = np.stack((top_k_indices.astype(np.float32), scores[top_k_indices].astype(np.float32)), axis=1)
    return candidates


# --- FUNCTION 2: Stages 2 & 3 Stochastic Search (FIXED) ---
def stochastic_search_ranking(user_id: str, candidates: np.ndarray, num_return: int = 10) -> List[str]:
    # ... (No functional changes needed here) ...
    global REDIS_CLIENT, ITEM_ID_TO_INDEX, ITEM_INDEX_TO_ID

    # 1. Get Live User Intent from Redis
    recent_items_ids = REDIS_CLIENT.lrange(f"user_history:{user_id}", 0, 9)
    recent_items_indices = [ITEM_ID_TO_INDEX.get(i) for i in recent_items_ids if i in ITEM_ID_TO_INDEX]

    # Filter candidates
    candidates_list = [(int(idx), score) for idx, score in candidates]
    filtered_candidates = [
        (idx, score) for idx, score in candidates_list
        if idx not in recent_items_indices
    ]

    if len(filtered_candidates) <= num_return:
        return [str(ITEM_INDEX_TO_ID.get(int(idx))) for idx, score in filtered_candidates]

    # Stochastic Logic (Exploitation/Exploration)
    filtered_candidates.sort(key=lambda x: x[1], reverse=True)
    num_exploit = int(num_return * 0.8)
    final_selection = []
    final_selection.extend([idx for idx, score in filtered_candidates[:num_exploit]])

    # ... (Exploration logic using numpy.random.choice) ...
    num_explore = num_return - num_exploit
    exploit_pool = filtered_candidates[:num_exploit * 2]
    explore_pool = filtered_candidates[num_exploit * 2:]

    if len(explore_pool) > 0:
        explore_indices = np.random.choice(len(explore_pool), size=min(num_explore, len(explore_pool)), replace=False)
        for i in explore_indices:
            final_selection.append(explore_pool[i][0])

    final_recommendations = [str(ITEM_INDEX_TO_ID.get(idx)) for idx in final_selection]
    np.random.shuffle(final_recommendations)

    return final_recommendations


# --- The Main API Endpoint (The Unified Router) ---
@app.get("/api/v1/predict", response_model=List[str])
def predict_recommendations(user_id: str,
                            mode: str = "CROSS_SELL",
                            recent_item_id: Optional[str] = None) -> List[str]:
    """
    The Unified Prediction Endpoint. Handles both trained users (CF/Hybrid)
    and Cold-Start users (Content-Based Fallback).
    """
    global USER_ID_TO_INDEX

    user_index = USER_ID_TO_INDEX.get(user_id)

    # --- ARCHITECTURAL ROUTING LOGIC ---

    # A. COLD START PATH (User ID is UNSEEN and has provided an item interest)
    if user_index is None and recent_item_id:
        print(f"Routing: COLD START for item {recent_item_id}")
        return cold_start_recommendations(recent_item_id)

    # B. UNSEEN USER / NO INTEREST PROVIDED
    if user_index is None:
        # User is brand new and didn't provide an item. Return error or top popular.
        raise HTTPException(
            status_code=404,
            detail="User ID not found and no recent item provided for cold start."
        )

    # C. HYBRID/TRAINED USER PATH
    # User is known, proceed with full Hybrid/Stochastic Search
    print(f"Routing: TRAINED USER HYBRID SEARCH for user {user_id}")

    # 2. Candidate Generation (Stage 1)
    candidates_with_scores = get_recommendations_and_scores(user_index, k=100)

    # 3. Stochastic Re-Ranking (Stages 2 & 3)
    final_recommendations = stochastic_search_ranking(user_id, candidates_with_scores, num_return=10)

    return final_recommendations