# Fashion Recommendation Engine

This project is a microservices-based system designed to simulate a real-world, data-driven recommendation engine for an e-commerce fashion storefront. It demonstrates core skills in distributed systems, data engineering, and machine learning operations (MLOps) by building a complete end-to-end pipeline.

The project is currently in its initial phase, with the foundational data ingestion pipeline fully implemented and a local development environment configured using Docker.

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

The following components have been successfully implemented and tested:

* **Database Schema:** A normalized relational schema for storing product and image data has been designed and is automatically created on startup using a PostgreSQL initialization script.

* **Data Ingestion Pipeline:** A Python script (`ingest.py`) has been developed to perform a batch load of 3,000 fashion items from a local CSV file into the PostgreSQL database.

* **Local Infrastructure:** A `docker-compose.yml` file is configured to spin up the entire development environment, including:

  * A PostgreSQL database with a persistent volume.

  * A Redis cache.

  * A Python-based `image_server` to serve local image files.

* **Version Control:** The project is under Git version control, with all core code and configurations committed.

## Next Steps

The next phase of the project will focus on building the real-time data pipeline and implementing the core recommendation logic. This includes:

* Developing a user interaction event generator.

* Building a stream processing service that consumes events from Kafka and updates the Redis cache.

* Implementing the content-based filtering and "complete the outfit" search algorithms.