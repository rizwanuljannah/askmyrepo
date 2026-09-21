from pymilvus import MilvusClient

from embed_repo import embed_all_chunks

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "code_chunks"

client = MilvusClient(uri=MILVUS_URI)
print("Connected to Milvus.")

# Get all the embedded chunks (clones repo, chunks, embeds — same as before)
records = embed_all_chunks()
print(f"\nPrepared {len(records)} record(s) to insert.")

# Insert them into the collection. Each record's keys (file_path,
# chunk_index, chunk_text, embedding) already match the schema fields —
# Milvus auto-generates the 'id' field itself.
result = client.insert(collection_name=COLLECTION_NAME, data=records)
print(f"\nInsert result: {result}")

# Milvus buffers new inserts before writing them to permanent storage,
# so we explicitly flush to make sure they're actually persisted before
# checking the count.
client.flush(COLLECTION_NAME)
print("Flushed to storage.")

# Verify by asking Milvus how many rows the collection actually holds now
stats = client.get_collection_stats(COLLECTION_NAME)
print(f"\nCollection now contains {stats['row_count']} row(s).")