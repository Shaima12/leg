"""
Test Rapide - Vérification du système RAG
==========================================
Teste rapidement toutes les composantes du pipeline
"""

from src.modules.embeddings import CodeTravailEmbedder
from src.modules.generation import CodeTravailGenerator, RAGPipeline


print("="*80)
print("  🧪 TEST RAPIDE - SYSTÈME RAG")
print("="*80)

# ============================================
# TEST 1: Connexion Qdrant
# ============================================
print("\n🔌 TEST 1: CONNEXION QDRANT")
print("-" * 80)

try:
    embedder = CodeTravailEmbedder()
    collections = embedder.qdrant.get_collections()
    print(f"✓ Connecté à Qdrant")
    print(f"✓ Collections disponibles: {[c.name for c in collections.collections]}")
    
    # Vérifier la collection
    collection_info = embedder.qdrant.get_collection(embedder.config.collection_name)
    print(f"✓ Collection '{embedder.config.collection_name}': {collection_info.points_count} points")
    
except Exception as e:
    print(f"❌ Erreur Qdrant: {e}")
    exit(1)

# ============================================
# TEST 2: Recherche Vectorielle
# ============================================
print("\n🔍 TEST 2: RECHERCHE VECTORIELLE")
print("-" * 80)

try:
    query_text = "durée du congé annuel"
    query_vec = embedder.embed_text(embedder.cleaner.clean(query_text))
    
    results = embedder.qdrant.query_points(
        collection_name=embedder.config.collection_name,
        query=query_vec,
        limit=3,
        with_payload=True
    )
    
    print(f"✓ Requête: '{query_text}'")
    print(f"✓ Résultats: {len(results.points)}\n")
    
    for i, hit in enumerate(results.points, 1):
        payload = hit.payload
        score = hit.score
        article = payload.get('article', 'N/A')
        print(f"   {i}. {article} (score: {score:.3f})")
    
except Exception as e:
    print(f"❌ Erreur recherche: {e}")
    exit(1)

# ============================================
# TEST 3: Génération LLM (Groq)
# ============================================
print("\n🤖 TEST 3: GÉNÉRATION LLM (GROQ)")
print("-" * 80)

try:
    generator = CodeTravailGenerator()
    print(f"✓ Générateur initialisé")
    print(f"✓ Modèle: {generator.config.model}")
    
except Exception as e:
    print(f"❌ Erreur générateur: {e}")
    exit(1)

# ============================================
# TEST 4: Pipeline RAG Complet
# ============================================
print("\n💬 TEST 4: PIPELINE RAG COMPLET")
print("-" * 80)

try:
    rag = RAGPipeline(embedder, generator)
    
    test_question = "Quelle est la durée du congé annuel?"
    print(f"📝 Question: {test_question}")
    
    result = rag.query(
        question=test_question,
        top_k=3,
        verbose=False
    )
    
    print("\n💡 RÉPONSE:")
    print("-" * 80)
    print(result['answer'])
    
    print("\n📚 SOURCES:")
    for i, ctx in enumerate(result['contexts'], 1):
        print(f"   {i}. {ctx['citation']} (score: {ctx['score']:.2%})")
    
    # Métriques
    metrics = generator.evaluate_response(result)
    print(f"\n📊 MÉTRIQUES:")
    print(f"   • Mots: {metrics['num_words']}")
    print(f"   • Sources: {metrics['num_sources']}")
    print(f"   • Citations: {'✓' if metrics['has_citations'] else '✗'}")
    
except Exception as e:
    print(f"❌ Erreur RAG: {e}")
    exit(1)

# ============================================
# RÉSUMÉ
# ============================================
print("\n" + "="*80)
print("✅ TOUS LES TESTS RÉUSSIS")
print("="*80)
print("\n💡 Le système est opérationnel!")
print("\n🚀 Pour lancer le pipeline complet:")
print("   python src/main_pipeline.py")
print("\n📖 Ou utilisez l'API:")
print("   from src.main_pipeline import CodeTravailAPI")
print("   api = CodeTravailAPI()")
print("   response = api.ask('Votre question ici')")