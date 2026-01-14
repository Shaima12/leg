import re
import unicodedata
from typing import List, Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
@dataclass
class EmbeddingConfig:
    collection_name: str = "code_travail_tunisien"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = 384
    batch_size: int = 100  # upload par batch pour éviter WriteTimeout

# -----------------------------
# NETTOYAGE DU TEXTE
# -----------------------------
class TextCleaner:
    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive

    def clean(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\f', '\n', text)
        if self.aggressive:
            text = re.sub(r'http[s]?://\S+', '[URL]', text)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
        return text.strip()

# -----------------------------
# CLIENT QDRANT CLOUD
# -----------------------------
qdrant = QdrantClient(
    url="https://81cb2705-ec2f-4b4e-9603-88368c5abb3d.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ynh283IPcSfyUZZp55FGDuqtG0sAoiMdfVW3UYF_iRQ",
    timeout=60  # Timeout plus long pour cloud
)

# -----------------------------
# EMBEDDER
# -----------------------------
class CodeTravailEmbedder:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.cleaner = TextCleaner()
        self.qdrant = qdrant
        print("✓ Connected to Qdrant Cloud")
        self.model = SentenceTransformer(self.config.model_name)
        print(f"✓ Loaded model ({self.config.vector_size} dimensions)")

    # Créer/recréer collection
    def create_collection(self, recreate: bool = False):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.config.collection_name in collections:
            if recreate:
                self.qdrant.delete_collection(self.config.collection_name)
                print("🗑️ Deleted existing collection")
            else:
                print("✓ Collection exists")
                return
        self.qdrant.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=self.config.vector_size, distance=Distance.COSINE)
        )
        print(f"🆕 Created collection: {self.config.collection_name}")

    # Nettoyer un chunk
    def clean_chunk(self, chunk: Dict) -> Dict:
        chunk['text'] = self.cleaner.clean(chunk['text'])
        return chunk

    # Générer embedding
    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    # Embed plusieurs chunks
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        embedded_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk = self.clean_chunk(chunk)
            chunk['embedding'] = self.embed_text(chunk['text'])
            embedded_chunks.append(chunk)
        return embedded_chunks

    # Upload vers Qdrant par batch
    def upload_chunks(self, chunks: List[Dict], recreate_collection: bool = False):
        self.create_collection(recreate=recreate_collection)
        if 'embedding' not in chunks[0]:
            chunks = self.embed_chunks(chunks)

        total_points = len(chunks)
        batch_size = self.config.batch_size
        print(f"📤 Upload {total_points} points par batchs de {batch_size}...")

        for start in range(0, total_points, batch_size):
            end = min(start + batch_size, total_points)
            batch_points = [
                PointStruct(
                    id=i,
                    vector=chunks[i]['embedding'],
                    payload=chunks[i]['metadata'] | {'text': chunks[i]['text']}
                )
                for i in range(start, end)
            ]
            self.qdrant.upsert(collection_name=self.config.collection_name, points=batch_points)
            print(f"   ✓ Uploaded points {start} à {end-1}")

        print(f"✅ Upload terminé: {total_points} chunks")

    # Recherche avec query_points pour tout le payload
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        query_vector = self.embed_text(self.cleaner.clean(query))
        results = self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        formatted_results = []
        for hit in results['result']:
            formatted_results.append({
                'score': hit['score'],
                'text': hit['payload'].get('text', ''),
                **hit['payload']
            })
        return formatted_results

# -----------------------------
# HELPER
# -----------------------------
def embed_and_upload(chunks: List[Dict], config: Optional[EmbeddingConfig] = None):
    embedder = CodeTravailEmbedder(config)
    embedder.upload_chunks(chunks, recreate_collection=True)
    return embedder

# -----------------------------
# EXEMPLE D'UTILISATION
# -----------------------------
if __name__ == "__main__":
    from chunking import CodeTravailChunker
    chunker = CodeTravailChunker()
    chunks = chunker.load_from_json("code_travail_chunks.json")
    embedder = CodeTravailEmbedder()
    embedder.upload_chunks(chunks, recreate_collection=True)

    queries = ["durée du congé annuel", "licenciement abusif", "salaire minimum"]
    for q in queries:
        results = embedder.search(q, limit=3)
        print(f"\n❓ Query: {q}")
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r['score']:.3f}, Text: {r['text'][:100]}...")
