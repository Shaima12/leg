from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://81cb2705-ec2f-4b4e-9603-88368c5abb3d.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.YZkAraSe0bf_4_985RToeoUx97E7mE6R3Yu1ULTdMTU",
)

print(qdrant_client.get_collections())