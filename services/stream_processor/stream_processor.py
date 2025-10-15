import os
import sys
import json
import redis
from confluent_kafka import Consumer, KafkaException

# Configuration for the Kafka Consumer
conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'group.id': 'stream-processor-group',
    'auto.offset.reset': 'earliest'  # Start reading from the beginning of the topic
}
consumer = Consumer(conf)

# Configuration for the Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST'),
    port=os.getenv('REDIS_PORT'),
    decode_responses=True  # Decodes responses to strings
)


def process_event(event):
    """
    Processes a single Kafka event and updates the Redis cache.
    """
    try:
        # We'll use the user ID as a key and store a list of recently viewed items
        user_id = event['customer_id']
        article_id = event['article_id']

        # Redis key for the user's recent history
        user_history_key = f"user_history:{user_id}"

        # Add the article_id to the user's history list
        # We use lpush to add to the left of the list
        redis_client.lpush(user_history_key, article_id)

        # Trim the list to a certain size (e.g., the 10 most recent items)
        redis_client.ltrim(user_history_key, 0, 9)

        print(f"Processed event for user {user_id}. Added article {article_id} to history.")

    except Exception as e:
        print(f"Error processing event: {e}")


if __name__ == '__main__':
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC')

    try:
        # Check Redis connection
        if redis_client.ping():
            print("Successfully connected to Redis.")
        else:
            print("Could not connect to Redis. Exiting.")
            sys.exit(1)

        # Subscribe to the Kafka topic
        consumer.subscribe([KAFKA_TOPIC])

        # Start the infinite loop to process messages
        while True:
            msg = consumer.poll(1.0)  # Poll for a message with a 1-second timeout

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    # End of partition event - not a real error
                    continue
                else:
                    print(f"Kafka error: {msg.error()}")
                    break

            # Decode the message and process it
            event_data = json.loads(msg.value().decode('utf-8'))
            process_event(event_data)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, shutting down.")
    finally:
        consumer.close()