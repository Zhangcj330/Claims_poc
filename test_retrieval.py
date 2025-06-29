import pytest
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


class TestRetrievalSystem:
    """Test suite for chunk ID validation."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_class_fixture(self, request):
        request.cls.embedder = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        request.cls.client = chromadb.PersistentClient(path="chromadb")
        request.cls.namespaces = [col.name for col in request.cls.client.list_collections()]

    def retrieve_contexts_for_question(self, question, namespace=None, k=10):
        """Helper method to retrieve contexts for a question."""
        all_docs = []
        
        # If no namespace specified, search all namespaces
        target_namespaces = [namespace] if namespace else self.namespaces

        for ns in target_namespaces:
            vectorstore = Chroma(
                client=self.client,
                collection_name=ns,
                embedding_function=self.embedder
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "lambda_mult": 0.8}
            )

            results = retriever.invoke(question)

            for doc in results:
                metadata = doc.metadata
                all_docs.append({
                    "namespace": ns,
                    "question": question,
                    "chunk_id": metadata.get("id", metadata.get("chunk_id", "N/A")),
                    "source_file": metadata.get("source_file", "N/A"),
                    "document_name": metadata.get("document_name", metadata.get("title", "N/A")),
                    "section_title": metadata.get("section_title", "N/A"),  
                    "subheading": metadata.get("subheading", "N/A"),
                    "content": doc.page_content,
                    "insurer": metadata.get("insurer", "N/A"),
                    "product_type": metadata.get("product_type", "N/A"),
                    "page_no": metadata.get("page_no", "N/A")
                })
        
        return all_docs

    def check_expected_chunk_ids(self, retrieved_docs, expected_chunk_ids):
        """
        Check if expected chunk IDs are present in the retrieved results.
        
        Args:
            retrieved_docs: List of retrieved document chunks
            expected_chunk_ids: List of chunk IDs that should be present
            
        Returns:
            dict: Results showing which chunk IDs were found and missing
        """
        found_chunk_ids = []
        missing_chunk_ids = []
        
        retrieved_chunk_ids = [doc['chunk_id'] for doc in retrieved_docs if doc['chunk_id'] != "N/A"]
        
        for expected_id in expected_chunk_ids:
            if expected_id in retrieved_chunk_ids:
                found_chunk_ids.append(expected_id)
            else:
                missing_chunk_ids.append(expected_id)
        
        return {
            'found_chunk_ids': found_chunk_ids,
            'missing_chunk_ids': missing_chunk_ids,
            'retrieved_chunk_ids': retrieved_chunk_ids,
            'total_chunks': len(retrieved_docs)
        }

    # ============================================================================
    # TEST DATA STRUCTURE - TO BE FILLED BY USER
    # ============================================================================
    
    @pytest.fixture
    def test_cases(self):
        """
        Define test cases with questions and expected chunk IDs.
        User should fill this with actual test data.
        """
        return [
            {
                "question": "what are the critical illness medical conditions applicable to Standard and Premier in TAL 2023",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "92", "97"
                ],
                "min_chunks_expected": 2,  # Minimum number of chunks expected
                "description": "Test description for question 1"
            },
            {
                "question": "What happens if policyholder die while covered under the policy",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "77", "24", "15"
                ],
                "min_chunks_expected": 3,  # Minimum number of chunks expected
                "description": "Chunk 77: Directly states the Death Benefit is paid. (The core answer)."
            },
            {
                "question": "Are there any exclusions to the Critical Illness cover?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "102"
                ],
                "min_chunks_expected": 1,  # Minimum number of chunks expected
                "description": "Chunk 102: Directly states the Critical Illness cover is excluded. (The core answer)."
            },
            {
                "question": "Can I get Income Protection if I'm self-employed?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "269"
                ],
                "min_chunks_expected": 1,  # Minimum number of chunks expected
                "description": "Chunk 269: Directly states the Income Protection is available for self-employed individuals. (The core answer)."
            },
            {
                "question": "What does Total and Permanent Disability (TPD) mean under this policy?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "275", "276","26"
                ],
                "min_chunks_expected": 3,  # Minimum number of chunks expected
                "description": "Test description for question 4"
            },
            {
                "question": "When does the policy end?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "82", "89", "203"
                ],
                "min_chunks_expected": 3,  # Minimum number of chunks expected
                "description": "Test description for question 5"
            },
            {
                "question": "What’s the waiting period before I can receive Income Protection payments?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "43"
                ],
                "min_chunks_expected": 1,  # Minimum number of chunks expected
                "description": "Test description for question 6"
            },
                    {
                "question": "How do I cancel the policy if I change my mind?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "14", "326", "201"
                ],
                "min_chunks_expected": 3,  # Minimum number of chunks expected
                "description": "Test description for question 7"
            },
                    {
                "question": "Can I hold the policy through my super fund?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "232", "66"
                ],
                "min_chunks_expected": 2,  # Minimum number of chunks expected
                "description": "Test description for question 8"
            },
            {
                "question": "How do I increase my cover if my circumstances change?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "108", "199"
                ],
                "min_chunks_expected": 2,  # Minimum number of chunks expected
                "description": "Test description for question 9"
            },
            {
                "question": "What are the premiums like?",
                "namespace": "TALR7983-0923-accelerated-protection-pds-8-sep-2023",  # Search all namespaces, or specify a specific one
                "expected_chunk_ids": [
                    "69", "70", "72"
                ],
                "min_chunks_expected": 3,  # Minimum number of chunks expected
                "description": "Test description for question 10"
            },
        ]

    # ============================================================================
    # ACTUAL TEST METHODS
    # ============================================================================
    
    @pytest.mark.parametrize("test_case", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Adjust indices based on number of test cases
    def test_retrieval_returns_expected_chunk_ids(self, test_cases, test_case):
        """Test that retrieval returns expected chunk IDs for a given question."""
        case = test_cases[test_case]
        
        # Skip test if no chunk IDs specified (template case)
        if not case["expected_chunk_ids"]:
            pytest.skip("No chunk IDs specified for this test case")
        
        # Retrieve contexts directly for the question
        retrieved_docs = self.retrieve_contexts_for_question(
            question=case["question"],
            namespace=case.get("namespace")
        )
        
        # Check chunk IDs
        chunk_results = self.check_expected_chunk_ids(
            retrieved_docs, 
            case["expected_chunk_ids"]
        )
        
        # Assertions
        assert len(retrieved_docs) >= case["min_chunks_expected"], \
            f"Expected at least {case['min_chunks_expected']} chunks, got {len(retrieved_docs)}"
        
        assert len(chunk_results["missing_chunk_ids"]) == 0, \
            f"Missing expected chunk IDs: {chunk_results['missing_chunk_ids']}"
        
        assert len(chunk_results["found_chunk_ids"]) == len(case["expected_chunk_ids"]), \
            f"Expected {len(case['expected_chunk_ids'])} chunk IDs, found {len(chunk_results['found_chunk_ids'])}"

    def test_retrieval_basic_functionality(self):
        """Test basic retrieval functionality without specific expectations."""
        test_query = "what are the critical illness conditions"
        
        # Retrieve contexts directly
        retrieved_docs = self.retrieve_contexts_for_question(test_query)
        
        # Basic assertions
        assert len(retrieved_docs) > 0, "No documents retrieved"
        
        # Check that each doc has required fields
        for doc in retrieved_docs:
            assert 'content' in doc, "Document missing content field"
            assert 'source_file' in doc, "Document missing source_file field"
            assert 'namespace' in doc, "Document missing namespace field"
            assert 'chunk_id' in doc, "Document missing chunk_id field"
            assert 'question' in doc, "Document missing question field"
            assert len(doc['content']) > 0, "Document content is empty"

    def test_chromadb_connection(self):
        """Test that ChromaDB connection is working."""
        assert self.client is not None, "ChromaDB client is None"
        assert len(self.namespaces) > 0, "No collections found in ChromaDB"
        
        # Test that we can access at least one collection
        test_collection = self.client.get_collection(self.namespaces[0])
        assert test_collection is not None, f"Cannot access collection {self.namespaces[0]}"


# ============================================================================
# EXAMPLE USAGE AND INSTRUCTIONS
# ============================================================================

"""
INSTRUCTIONS FOR FILLING TEST DATA:

1. Modify the test_cases fixture to include your actual test questions
2. For each test case, provide:
   - question: The actual question to test
   - namespace: Specific namespace to search (optional, None = search all)
   - expected_chunk_ids: List of chunk IDs that should be returned for this question
   - min_chunks_expected: Minimum number of chunks expected to be retrieved
   - description: Human-readable description of what this test case covers

Example filled test case:
{
    "question": "what are the critical illness medical conditions applicable to Standard and Premier in TAL 2023",
    "namespace": "tal_2023_collection",  # Optional: specify which collection to search
    "expected_chunk_ids": [
        "tal_2023_chunk_001", "tal_2023_chunk_045", "tal_2023_chunk_156"
    ],
    "min_chunks_expected": 3,
    "description": "Test retrieval of critical illness conditions for TAL 2023 products"
}

3. Update the @pytest.mark.parametrize decorators to match the number of test cases you have

4. Run tests with:
   pytest test_retrieval.py -v                                                # Run all tests
   pytest test_retrieval.py::TestRetrievalSystem::test_retrieval_returns_expected_chunk_ids -v  # Run specific test

""" 