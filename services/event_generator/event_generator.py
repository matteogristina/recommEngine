import os
import sys
import json
import psycopg2
from confluent_kafka import Producer

# Configuration for the Kafka Producer
conf = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'client.id': 'event-generator'
}
producer = Producer(conf)

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using environment variables.
    Returns: A database connection object.
    """
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        sys.exit(1)

def delivery_report(err, msg):
    """Called once for each message produced to indicate delivery result."""
    if err is not None:
        print(f"Message delivery failed: {err}")


def produce_events(prod, topic, connect):
    """
    Reads transaction data from the database and publishes each row to a Kafka topic.
    """
    cur = connect.cursor()
    print("Starting to read transactions from the database...")

    test_query = """
            SELECT t_dat, customer_id, article_id, price
            FROM transactions
            WHERE t_dat < '2019-01-01'
            ORDER BY t_dat ASC;
            """

    query = """
            SELECT t_dat, customer_id, article_id, price
            FROM transactions
            ORDER BY t_dat ASC;
            """
    cur.execute(query)

    print("Starting to produce events to Kafka...")
    processed_count = 0

    # Use a loop that iterates directly over the cursor
    for row in cur:
        event = {
            't_dat': row[0].isoformat(),
            'customer_id': row[1],
            'article_id': str(row[2]),
            'price': float(row[3])
        }
        event_json = json.dumps(event).encode('utf-8')

        # We need a loop here to handle the "Queue full" error gracefully.
        while True:
            try:
                # Attempt to produce the message
                prod.produce(topic, key=row[1], value=event_json, callback=delivery_report)
                break  # Break out of the while loop if produce is successful
            except BufferError:
                # If the queue is full, block for up to 1 second to wait for space
                print('Local: Queue full. Waiting for space...')
                prod.poll(1)

        processed_count += 1
        if processed_count % 100000 == 0:
            print(f"Processed {processed_count} messages.")

    # Wait for any outstanding messages to be delivered
    prod.flush()
    print(f"All {processed_count} events produced successfully!")
    cur.close()

if __name__ == "__main__":
    conn = None
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC')
    try:
        conn = get_db_connection()
        # Corrected function call with the proper arguments
        produce_events(producer, KAFKA_TOPIC, conn)

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn: conn.close()
        sys.exit(1)