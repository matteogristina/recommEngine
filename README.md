# Fashion Recommendation Engine

This project is a microservices-based system designed to simulate a real-world, data-driven recommendation engine for an e-commerce fashion storefront. It demonstrates core skills in distributed systems, data engineering, and machine learning operations (MLOps) by building a complete end-to-end pipeline.

The project has successfully completed its foundational data ingestion phase and is now a robust, Docker-based environment ready to process real-time user data.

## Project Goals

The primary goal of this project is to build an intelligent recommendation system from the ground up, showcasing a multi-faceted skill set required for a modern developer role. The key features to be implemented are:

* **Content-Based Filtering:** Recommending items based on similarities in their attributes (e.g., brand, category, description).

* **"Complete the Outfit" Search:** An advanced search feature that suggests complementary items to a user's shopping cart.

* **Real-time Recommendations:** Providing dynamic recommendations based on a user's recent activity.

## Architecture & Technology Stack

The system is built on an event-driven, containerized architecture.

* **Orchestration:** Docker Compose

* **Databases:**

  * **PostgreSQL:** Serves as the primary, persistent database for the product catalog.

  * **Redis:** Will be used as an in-memory, low-latency store for real-time features and user activity.

* **Messaging Queue:** Kafka (for real-time event streaming)

* **Data Ingestion:** A Python script for batch loading data into PostgreSQL.

* **Images:** A local web server to serve image files for the storefront mock.

## Current Progress (Phase 1: Planning & Data Layer)

The following components have been successfully implemented, tested, and pushed to GitHub:

* **Database Schema:** A normalized relational schema for a real-world dataset has been designed and is automatically created on startup. This includes dedicated tables for products, articles, customers, and a massive transaction history.

* **Data Ingestion Pipeline:** A robust Python script (`ingest.py`) has been developed to handle the complexities of a large, real-world dataset. The pipeline:
    * Processes and ingests data from three separate CSVs (100k articles, 1.3m customers, and over 31m transactions).
    * Performs on-the-fly data cleaning and type-casting to handle inconsistencies.
    * Uses a memory-efficient chunking method to ingest the multi-gigabyte transactions file without running out of memory.

* **Local Infrastructure:** A `docker-compose.yml` file is configured to spin up the entire development environment, including:
    * A PostgreSQL database with a persistent volume for data durability.
    * A Redis cache.
    * A Kafka broker and Zookeeper for message streaming.
    * A Python-based `image_server` to serve local image files.

* **Version Control:** All core code, configurations, and environment setup have been committed to the Git repository, ensuring the project is fully reproducible.

## Next Steps (Phase 2: User Interaction & Real-time Pipeline)

The next phase of the project will focus on building the real-time data pipeline to simulate live user activity. This includes:

* **Event Generator:** Developing a Python service that replays the historical transaction data in chronological order and publishes it as a live event stream to Kafka.

* **Stream Processor:** Building a new service that consumes these real-time events, processes them, and aggregates features for a real-time feature store.

* **Real-time Feature Store:** Implementing the logic to store and retrieve aggregated user behavior data from Redis for use in the recommendation service.