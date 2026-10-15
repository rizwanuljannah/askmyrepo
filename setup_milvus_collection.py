from dotenv import load_dotenv
load_dotenv()
import os

from pymilvus import MilvusClient, DataType

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
COLLECTION_NAME = "code_chunks"
EMBEDDING_DIM = 768  # matches nomic-embed-text

client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
print("Connected to Milvus.")

# Drop the collection first if it already exists, so this script can be re-run cleanly
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"Dropped existing collection '{COLLECTION_NAME}'.")

# Define the schema: auto id, file_path, chunk_index, chunk_text, and the embedding vector
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

# Index settings: COSINE similarity with AUTOINDEX (Zilliz/Milvus picks the best index type for you)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

# Create the collection with this schema and index
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)

print(f"Collection '{COLLECTION_NAME}' created with schema and index.")

# Confirm it's ready
stats = client.get_collection_stats(COLLECTION_NAME)
print(f"Collection stats: {stats}")