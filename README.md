# Real-Time Event Processing & Hybrid AI Search Platform

This project is a microservices-based system designed to simulate a high-volume, production-grade recommendation engine. It serves as a comprehensive demonstration of expertise in **Real-Time Distributed Systems, MLOps, and Advanced AI Search Algorithms.**

The system is fully deployed via Docker and currently hosts a trained Hybrid Recommendation Model ready for low-latency inference.

## Project Goals (MLOps & Search Focus)

The core objective is to build an intelligent, scalable serving platform from the ground up, showcasing specialized skills required for a modern developer role.

* **Hybrid Prediction:** Combining long-term user taste (Collaborative Filtering) with immediate product attributes (Content-Based Filtering) to solve the **Cold-Start Problem**.
* **Stochastic Search:** Implementing a ranking strategy that maximizes both conversion (exploitation) and product discovery (exploration).
* **Low-Latency Serving:** Delivering predictions via a dedicated FastAPI microservice, leveraging Redis for sub-10ms feature lookups.

## Architecture & Technology Stack

The system is built on an event-driven, containerized architecture designed for performance and horizontal scalability.

* **Orchestration:** Docker Compose
* **Serving:** **FastAPI** (Low-latency ML microservice)
* **Databases:**
    * **PostgreSQL:** Persistent storage for the product catalog and historical data.
    * **Redis:** **High-speed Feature Store** for real-time user intent signals (e.g., last 10 viewed items).
* **Messaging Queue:** Kafka (Durable log for real-time event streaming)

## Current Progress (Phase 3: Model Deployment Complete)

**The entire ML serving pipeline is complete and operational.** The following components have been successfully implemented and tested:

* **Data Engineering Pipeline:**
    * Robust ingestion of **37 Million** transactions, **1.3 Million** customers, and **100k** products.
    * Memory-efficient **chunking logic** to handle multi-gigabyte files without crashing (solving the common OOM bottleneck).

* **Real-Time Feature Generation:**
    * The **Event Generator** successfully streams all 37M historical transactions to Kafka (simulating live user activity).
    * The **Stream Processor** runs in parallel, consuming events and populating the **Redis Feature Store** with a sliding window of the user's 10 most recent interactions.

* **Hybrid Model Training & Deployment (The MLOps Deliverable):**
    * **Model Training:** A **Hybrid Collaborative/Content Filtering Model** was trained using a temporal split, generating dense **Item and User Embedding Vectors**.
    * **Artifact Persistence:** The trained model and all necessary mappings (`.joblib`, `.npy` files) are saved as versioned artifacts to a dedicated volume.

* **Low-Latency Recommendation API (Deployment):**
    * A **FastAPI Microservice** is deployed, which loads the ML artifacts at startup.
    * The API is ready to serve predictions, demonstrating model deployment experience.

## Next Steps (Phase 4: Validation & Finalization)

The next phase will focus on testing the complexity of the deployed model and creating a compelling demonstrator for recruiters. This includes:

* **Final API Logic:** Implementing the `COLD_START` and `SIMILARITY` search modes (Cosine Similarity against item embeddings).
* **Validation:** Creating the final test script to benchmark the model's MAP@12 performance against the reserved test set.
* **Service Demonstrator:** Building a simple web interface to visually showcase the generalized API's functions (`CROSS_SELL`, `SIMILARITY`, and `REAL-TIME RECENCY`).