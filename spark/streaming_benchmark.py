import sys
import time
import random
from pyspark import SparkContext
from collections import deque
from pyspark.streaming import StreamingContext

def main():
    """
    A PySpark DStreams (classic Streaming) benchmark application.

    This uses the classic Spark Streaming API (DStreams) to provide:
    - The classic Streaming tab in Spark UI
    - Processing Time, Scheduling Delay, and Total Delay metrics
    - Input/Processing rate visualization

    The workload scales across 30 executors with 5-7GB each.
    """
    if len(sys.argv) > 2:
        print("Usage: streaming_benchmark.py [batchIntervalSeconds]", file=sys.stderr)
        sys.exit(-1)

    # Batch interval in seconds - how often to process micro-batches
    batch_interval = int(sys.argv[1]) if len(sys.argv) == 2 else 5

    print(f"Starting DStreams benchmark with {batch_interval}s batch interval")

    # Create Spark Context
    sc = SparkContext(appName="StreamingBenchmark")
    sc.setLogLevel("WARN")

    # Create StreamingContext with batch interval
    ssc = StreamingContext(sc, batch_interval)
    # Note: Checkpointing is disabled since we don't have a distributed filesystem
    # For stateful operations, we'll use in-memory state

    # Create a queue of RDDs to simulate streaming data
    # This will be the input stream
    input_queue = deque(maxlen=10)

    def generate_batch_data():
        """Generate synthetic data for each batch"""
        # Generate 2 million events per batch - smaller to avoid driver OOM
        num_events = 2000000
        events = []
        for i in range(num_events):
            event_id = random.randint(1, 1000000)
            user_id = f"user_{event_id % 10000}"
            category = f"cat_{event_id % 50}"
            amount = random.uniform(1.0, 1000.0)
            events.append((event_id, user_id, category, amount))
        return events

    # Pre-populate queue with initial batch (required by queueStream)
    print("Generating initial batch...")
    initial_batch = generate_batch_data()
    # Use 60 partitions (2 per executor) to ensure more memory per task
    input_queue.append(sc.parallelize(initial_batch, 60))
    print(f"Initial batch ready with {len(initial_batch)} events")

    # Clear the initial batch from memory
    del initial_batch

    # Function to continuously add batches to the queue
    def add_batch_to_queue():
        """Add a new batch every interval, limiting queue size"""
        while True:
            time.sleep(batch_interval)
            # Limit queue size to prevent driver OOM
            # Only add new batch if queue has been processed
            if len(input_queue) < 3:
                batch_data = generate_batch_data()
                rdd = sc.parallelize(batch_data, 60)  # 60 partitions (2 per executor)
                input_queue.append(rdd)
                # Clear batch_data from driver memory immediately
                del batch_data
                print(f"Added batch to queue (queue size: {len(input_queue)})")
            else:
                print(f"Queue full ({len(input_queue)} batches), waiting for processing...")

    # Start thread to generate batches
    import threading
    batch_generator = threading.Thread(target=add_batch_to_queue, daemon=True)
    batch_generator.start()

    # Create input stream from queue
    input_stream = ssc.queueStream(input_queue, oneAtATime=True)

    def process_partition(partition):
        """
        Process each partition with CPU and memory-intensive operations.
        Gradually builds up to 2.5GB per task to achieve 5-6GB per executor.
        """
        # Allocate 2GB working memory per task - but don't pre-fill it
        memory_size_gb = 2
        memory_size_bytes = memory_size_gb * 1024 * 1024 * 1024

        print(f"EXECUTOR: Allocating {memory_size_gb}GB bytearray for processing...")
        # Just allocate, don't pre-fill to avoid initialization spike
        working_memory = bytearray(memory_size_bytes)

        print(f"EXECUTOR: Starting data processing with gradual memory fill...")

        # Build lookup table gradually during processing to avoid spike
        # Each entry is 20KB, we'll build up to 50K entries = 1GB total
        lookup_table = {}

        results = []
        record_count = 0
        partition_list = list(partition)
        total_records = len(partition_list)

        for event_id, user_id, category, amount in partition_list:
            # CPU-intensive enrichment
            enriched_amount = amount
            for i in range(100):
                enriched_amount = (enriched_amount * 1.001 + random.random()) % 10000

            # Memory-intensive operation - access and fill working memory gradually
            processing_score = 0.0
            for i in range(100):
                mem_idx = int((event_id + i) % memory_size_bytes)
                # Fill memory as we access it
                if working_memory[mem_idx] == 0:
                    working_memory[mem_idx] = random.randint(1, 255)
                working_memory[mem_idx] ^= (event_id & 0xFF)
                processing_score += working_memory[mem_idx]

            # Gradually build lookup table - add entry every N records
            # This spreads the 1GB allocation over time
            lookup_key = f"key_{event_id % 50000}"
            if lookup_key not in lookup_table:
                # Each entry is 20KB instead of 10KB for more memory
                lookup_table[lookup_key] = bytearray(20 * 1024)

            # Access the lookup table
            lookup_table[lookup_key][0] ^= (event_id & 0xFF)
            processing_score += lookup_table[lookup_key][0]

            processing_score = processing_score / 100

            results.append((category, (enriched_amount, processing_score, 1)))
            record_count += 1

            # Periodically fill more of the working memory to keep usage high
            if record_count % 5000 == 0:
                # Fill a 10MB chunk
                chunk_start = (record_count * 10000) % (memory_size_bytes - 10 * 1024 * 1024)
                for j in range(chunk_start, chunk_start + 10 * 1024 * 1024, 4096):
                    working_memory[j] = random.randint(1, 255)

        # Final memory report
        lookup_size_mb = len(lookup_table) * 20 / 1024  # 20KB per entry
        print(f"EXECUTOR: Processed {record_count} records, lookup table: {len(lookup_table)} entries ({lookup_size_mb:.1f}MB)")

        return iter(results)

    # Process the stream with CPU and memory intensive operations
    processed_stream = input_stream.mapPartitions(process_partition)

    # Reduce by key to aggregate per category
    aggregated_stream = processed_stream.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    )

    # Calculate averages and format output
    def format_results(rdd):
        """Format and print results for each batch"""
        results = rdd.collect()
        if results:
            print("\n" + "=" * 80)
            print(f"Batch Results - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print(f"{'Category':<15} {'Count':>10} {'Avg Amount':>15} {'Avg Score':>15}")
            print("-" * 80)
            for category, (total_amount, total_score, count) in results[:10]:
                avg_amount = total_amount / count
                avg_score = total_score / count
                print(f"{category:<15} {count:>10} {avg_amount:>15.2f} {avg_score:>15.2f}")
            print("=" * 80)
            print(f"Total categories processed: {len(results)}")
            print("=" * 80 + "\n")

    aggregated_stream.foreachRDD(format_results)

    # Add windowed counts for additional metrics (doesn't require checkpointing)
    category_counts = processed_stream.map(lambda x: (x[0], x[1][2]))
    windowed_counts = category_counts.window(batch_interval * 3, batch_interval)

    # Print windowed counts periodically
    def print_windowed(rdd):
        """Print windowed counts"""
        if not rdd.isEmpty():
            results = rdd.reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False).take(10)
            if results:
                print("\n" + "-" * 80)
                print(f"Windowed Event Counts by Category (Last {batch_interval * 3}s, Top 10)")
                print("-" * 80)
                for category, count in results:
                    print(f"{category:<15} {count:>10}")
                print("-" * 80 + "\n")

    windowed_counts.foreachRDD(print_windowed)

    print("=" * 80)
    print("DStreams benchmark is running!")
    print(f"Batch interval: {batch_interval} seconds")
    print(f"Processing across 30 executors with 5-7GB each")
    print("Check the Spark UI:")
    print("  - Streaming tab for Processing Time, Scheduling Delay, Total Delay")
    print("  - Input Rate and Processing Rate graphs")
    print("  - Executors tab for memory usage")
    print("=" * 80)

    # Start the streaming context
    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":
    main()
