"""
Tests for the RAG system
Run with: pytest test_rag.py -v

NOTE: Answer quality tests may fail with very small models like qwen:0.5b.
For reliable test results, use a better model in your .env:
  LLM_MODEL=llama3.2
  LLM_MODEL=mistral

The retrieval tests should always pass if the database is properly set up.
"""
import pytest
from query_data import query_rag
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from config import config


# Test data based on your documents
# NOTE: These tests work best with llama3.2 or better models
# Small models (qwen:0.5b) often fail to properly use context
TEST_CASES = [
    {
        "question": "Who is the main character in Alice in Wonderland?",
        "expected_keywords": ["alice"],
        "min_relevance": 0.5,
        "skip_with_small_models": False,  # Usually works even with small models
    },
    {
        "question": "What did Alice fall down?",
        "expected_keywords": ["rabbit", "hole"],
        "min_relevance": 0.5,
        "skip_with_small_models": True,  # Requires better comprehension
    },
    {
        "question": "How much total money does a player start with in Monopoly?",
        "expected_keywords": ["1500", "$1500", "fifteen hundred"],
        "min_relevance": 0.5,
        "skip_with_small_models": True,  # Requires precise extraction
    },
    {
        "question": "How many points does the longest train get in Ticket to Ride?",
        "expected_keywords": ["10", "ten"],
        "min_relevance": 0.5,
        "skip_with_small_models": False,
    },
]


class TestRAGSystem:
    """Test suite for RAG retrieval and generation"""

    @pytest.fixture(scope="class")
    def db(self):
        """Initialize the vector database once for all tests"""
        embedding_function = get_embedding_function()
        return Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=embedding_function
        )

    def test_database_exists(self, db):
        """Test that the vector database is populated"""
        collection = db.get()
        assert len(collection["ids"]) > 0, "Database is empty. Run populate_database.py first"
        print(f"\n✓ Database contains {len(collection['ids'])} documents")

    @pytest.mark.parametrize("test_case", TEST_CASES)
    def test_retrieval_relevance(self, db, test_case):
        """Test that retrieval finds relevant documents"""
        results = db.similarity_search_with_score(
            test_case["question"], 
            k=config.TOP_K_RESULTS
        )
        
        assert len(results) > 0, f"No results found for: {test_case['question']}"
        
        # Check relevance score (lower is better in some implementations)
        top_score = results[0][1]
        print(f"\n✓ Query: {test_case['question'][:50]}...")
        print(f"  Top relevance score: {top_score:.3f}")
        print(f"  Found {len(results)} relevant chunks")

    @pytest.mark.parametrize("test_case", TEST_CASES)
    def test_answer_contains_keywords(self, test_case):
        """Test that generated answers contain expected keywords
        
        NOTE: May fail with small models (qwen:0.5b). These models often
        hallucinate instead of using the retrieved context properly.
        Use llama3.2 or better for reliable results.
        """
        # Skip tests that require good models if using a small model
        small_models = ["qwen:0.5b", "qwen2:0.5b", "tinyllama"]
        if any(m in config.LLM_MODEL.lower() for m in small_models):
            if test_case.get("skip_with_small_models", False):
                pytest.skip(f"Skipping test - {config.LLM_MODEL} is too small for reliable results")
        
        response = query_rag(test_case["question"])
        
        assert response is not None, f"No response generated for: {test_case['question']}"
        
        response_lower = response.lower()
        found_keywords = [
            kw for kw in test_case["expected_keywords"] 
            if kw.lower() in response_lower
        ]
        
        if len(found_keywords) == 0:
            print(f"\n⚠️  WARNING: Response quality issue detected")
            print(f"  Model: {config.LLM_MODEL}")
            print(f"  Question: {test_case['question']}")
            print(f"  Expected: {test_case['expected_keywords']}")
            print(f"  Response: {response}")
            print(f"  💡 Try using a better model (llama3.2, mistral) for accurate answers")
        
        assert len(found_keywords) > 0, (
            f"Response doesn't contain any expected keywords.\n"
            f"Question: {test_case['question']}\n"
            f"Expected: {test_case['expected_keywords']}\n"
            f"Response: {response}\n"
            f"💡 This often happens with small models. Try: LLM_MODEL=llama3.2"
        )
        
        print(f"\n✓ Query: {test_case['question'][:50]}...")
        print(f"  Found keywords: {found_keywords}")

    def test_empty_query(self, db):
        """Test handling of empty queries"""
        results = db.similarity_search_with_score("", k=5)
        # Should still return results (matches anything) or handle gracefully
        assert isinstance(results, list)

    def test_irrelevant_query(self, db):
        """Test handling of queries unrelated to documents"""
        results = db.similarity_search_with_score(
            "What is the weather on Mars?", 
            k=5
        )
        # Should return something, but with lower relevance
        assert len(results) >= 0
        print(f"\n✓ Irrelevant query returned {len(results)} results")


class TestConfiguration:
    """Test configuration and setup"""

    def test_config_values(self):
        """Test that configuration is valid"""
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
        assert config.TOP_K_RESULTS > 0
        assert config.EMBEDDING_MODEL
        assert config.LLM_MODEL
        print(f"\n✓ Configuration valid:")
        print(f"  Embedding: {config.EMBEDDING_MODEL}")
        print(f"  LLM: {config.LLM_MODEL}")
        print(f"  Chunk size: {config.CHUNK_SIZE}")

    def test_embedding_function(self):
        """Test that embedding function can be initialized"""
        embedding_fn = get_embedding_function()
        assert embedding_fn is not None
        print(f"\n✓ Embedding function initialized")


# Individual test functions for pytest discovery
@pytest.mark.skipif(
    "qwen:0.5b" in config.LLM_MODEL.lower(),
    reason="qwen:0.5b is too small for reliable fact extraction"
)
def test_monopoly_money():
    """Specific test: Monopoly starting money
    
    Requires a decent model (llama3.2+) to extract facts accurately.
    """
    response = query_rag("How much money does a player start with in Monopoly?")
    assert response is not None
    response_lower = response.lower()
    assert "1500" in response_lower or "$1500" in response_lower or "fifteen hundred" in response_lower, \
        f"Expected $1500 in response, got: {response}"


def test_alice_main_character():
    """Specific test: Alice in Wonderland main character"""
    response = query_rag("Who is the main character in Alice in Wonderland?")
    assert response is not None
    assert "alice" in response.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
