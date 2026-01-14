import os
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from groq import Groq

# -----------------------------
# CONFIGURATION
# -----------------------------
@dataclass
class GeneratorConfig:
    """Configuration pour le générateur ChatGroq"""
    model_name: str = "llama-3.3-70b-versatile"  # Options: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
    temperature: float = 0.1  # Basse température pour réponses factuelles
    max_tokens: int = 1024
    top_p: float = 0.9
    stream: bool = False  # Active le streaming des réponses

# -----------------------------
# TEMPLATES DE PROMPTS
# -----------------------------
class PromptTemplates:
    """Templates de prompts pour différents cas d'usage"""
    
    LEGAL_QA = """Tu es un assistant juridique expert en droit du travail tunisien. Ta mission est de répondre aux questions en te basant UNIQUEMENT sur les extraits du Code du Travail Tunisien fournis.

**CONTEXTE (Code du Travail Tunisien):**
{context}

**QUESTION:**
{question}

**INSTRUCTIONS:**
1. Réponds UNIQUEMENT en te basant sur le contexte fourni
2. Si l'information n'est pas dans le contexte, dis-le clairement
3. Cite les articles pertinents dans ta réponse
4. Sois précis et factuel
5. Utilise un langage clair et accessible
6. Structure ta réponse avec des paragraphes si nécessaire

**RÉPONSE:**"""

    
# -----------------------------
# GÉNÉRATEUR PRINCIPAL
# -----------------------------
class CodeTravailGenerator:
    """
    Générateur de réponses utilisant ChatGroq pour le RAG sur le Code du Travail
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        config: Optional[GeneratorConfig] = None
    ):
        """
        Args:
            api_key: Clé API Groq (ou via variable d'environnement GROQ_API_KEY)
            config: Configuration personnalisée
        """
        self.config = config or GeneratorConfig()
        
        # Initialisation du client Groq
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Clé API Groq manquante. Fournis-la en paramètre ou via GROQ_API_KEY"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.templates = PromptTemplates()
        
        print(f"✓ Generator initialized with model: {self.config.model_name}")
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        template_type: str = "qa"
    ) -> str:
        """Construit le prompt selon le type demandé"""
        
        templates_map = {
            "qa": self.templates.LEGAL_QA
        }
        
        template = templates_map.get(template_type, self.templates.LEGAL_QA)
        return template.format(context=context, question=question)
    
    def generate(
        self,
        question: str,
        context: str,
        template_type: str = "qa",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None
    ) -> Dict:
        """
        Génère une réponse basée sur le contexte fourni
        
        Args:
            question: Question de l'utilisateur
            context: Contexte extrait (résultats du retrieval)
            template_type: Type de prompt à utiliser
            temperature: Température de génération (override config)
            max_tokens: Nombre max de tokens (override config)
            stream: Active le streaming (override config)
            
        Returns:
            Dict avec 'answer', 'prompt', 'usage', etc.
        """
        # Paramètres
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        stream = stream if stream is not None else self.config.stream
        
        # Construction du prompt
        prompt = self._build_prompt(question, context, template_type)
        
        # Appel à l'API Groq
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Tu es un assistant juridique expert en droit du travail tunisien."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            if stream:
                return self._generate_stream(messages, temperature, max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.top_p
            )
            
            return {
                'answer': response.choices[0].message.content,
                'prompt': prompt,
                'model': self.config.model_name,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason
            }
            
        except Exception as e:
            return {
                'answer': None,
                'error': str(e),
                'prompt': prompt
            }
    
    def _generate_stream(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ):
        """Génération en streaming"""
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config.top_p,
            stream=True
        )
        
        return {
            'stream': stream,
            'model': self.config.model_name
        }
    
    def generate_with_citations(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        max_context_length: int = 3000
    ) -> Dict:
        """
        Génère une réponse avec citations automatiques
        
        Args:
            question: Question utilisateur
            retrieved_chunks: Liste de chunks du retrieval (avec metadata)
            max_context_length: Longueur max du contexte
        """
        # Construction du contexte avec numérotation
        context_parts = []
        sources_map = {}
        current_length = 0
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk['text']
            article = chunk.get('metadata', {}).get('article', 'N/A')
            
            chunk_text = f"[Source {i} - {article}]\n{text}\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_context_length:
                break
            
            context_parts.append(chunk_text)
            sources_map[i] = {
                'article': article,
                'text': text,
                'score': chunk.get('score', 0)
            }
            current_length += chunk_length
        
        context = "\n".join(context_parts)
        
        # Génération
        result = self.generate(question, context, template_type="qa")
        
        # Ajout des sources
        result['sources'] = sources_map
        result['num_sources'] = len(sources_map)
        
        return result
    
# -----------------------------
# CLASSE UTILITAIRE DE RÉPONSE
# -----------------------------
class GeneratorResponse:
    """Encapsule et formate les réponses du générateur"""
    
    def __init__(self, response_dict: Dict):
        self.raw = response_dict
        self.answer = response_dict.get('answer')
        self.error = response_dict.get('error')
        self.usage = response_dict.get('usage', {})
        self.sources = response_dict.get('sources', {})
    
    def print_response(self):
        """Affiche la réponse formatée"""
        print("\n" + "=" * 70)
        print("📝 RÉPONSE")
        print("=" * 70)
        
        if self.error:
            print(f"❌ Erreur: {self.error}")
            return
        
        print(self.answer)
        
        if self.sources:
            print("\n" + "─" * 70)
            print("📚 SOURCES UTILISÉES")
            print("─" * 70)
            for source_id, source_info in self.sources.items():
                article = source_info['article']
                score = source_info.get('score', 0)
                print(f"[{source_id}] {article} (Pertinence: {score:.2f})")
        
        if self.usage:
            print("\n" + "─" * 70)
            print("📊 UTILISATION")
            print("─" * 70)
            print(f"Tokens prompt: {self.usage.get('prompt_tokens', 0)}")
            print(f"Tokens réponse: {self.usage.get('completion_tokens', 0)}")
            print(f"Total: {self.usage.get('total_tokens', 0)}")
    
    def to_dict(self) -> Dict:
        """Retourne la réponse en dict"""
        return self.raw
    
    def get_answer(self) -> Optional[str]:
        """Retourne uniquement la réponse"""
        return self.answer

# -----------------------------
# EXEMPLE D'UTILISATION
# -----------------------------
if __name__ == "__main__":
    import sys
    
    # EXEMPLE 1: Génération simple avec contexte
    print("\n" + "=" * 70)
    print("EXEMPLE 1: Génération simple")
    print("=" * 70)
    
    context = """
    [Source 1 - Article 114]
    La durée normale du travail des travailleurs occupés dans les secteurs non agricoles 
    est fixée à quarante heures par semaine et à 2288 heures par an.
    
    [Source 2 - Article 115]
    Dans les secteurs agricoles et forestiers, la durée normale du travail est fixée à 
    2496 heures par an.
    """
    
    question = "Quelle est la durée légale du travail par semaine?"
    
    result = generator.generate(question, context)
    response = GeneratorResponse(result)
    response.print_response()
    
    # EXEMPLE 2: Génération avec citations
    print("\n" + "=" * 70)
    print("EXEMPLE 2: Génération avec citations automatiques")
    print("=" * 70)
    
    # Simulation de chunks récupérés
    mock_chunks = [
        {
            'text': "Le congé annuel est fixé à un jour par mois de travail effectif.",
            'metadata': {'article': 'Article 113'},
            'score': 0.92
        },
        {
            'text': "La période de congé ne peut être fractionnée.",
            'metadata': {'article': 'Article 114'},
            'score': 0.85
        }
    ]
    
    question2 = "Combien de jours de congé ai-je droit par an?"
    result2 = generator.generate_with_citations(question2, mock_chunks)
    response2 = GeneratorResponse(result2)
    response2.print_response()
    
    # EXEMPLE 3: Explication d'article
    print("\n" + "=" * 70)
    print("EXEMPLE 3: Explication d'article")
    print("=" * 70)
    
    article_text = """
    Article 23: L'employeur est tenu d'assurer aux travailleurs des conditions de travail 
    décentes et de veiller à l'application des mesures de sécurité et d'hygiène.
    """
    
    result3 = generator.explain_article(article_text)
    response3 = GeneratorResponse(result3)
    response3.print_response()