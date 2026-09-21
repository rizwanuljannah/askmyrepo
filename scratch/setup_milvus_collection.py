from pymilvus import MilvusClient, DataType

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "code_chunks"
EMBEDDING_DIM = 768  # must match nomic-embed-text's output size

client = MilvusClient(uri=MILVUS_URI)
print("Connected to Milvus.")

if client.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists.")
else:
    # Define the "shape" of each record we'll store
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    # Tell Milvus how to index the embedding field for fast searching
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",  # a standard way to measure "how similar are two vectors"
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}'.")

# Print back the collection's structure as a sanity check
info = client.describe_collection(COLLECTION_NAME)
print("\nCollection info:")
print(info)