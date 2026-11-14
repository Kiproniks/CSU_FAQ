import unittest
from unittest.mock import patch

import chromadb

from ChunkBased.ChunkBased import ChunkBased

class TestChunkBased(unittest.TestCase):
    
    def setUp(self):
        with patch("ChunkBased.ChunkBased.chromadb.PersistentClient", return_value=chromadb.EphemeralClient()):
            self.chunk_system = ChunkBased(
                chunk_size=50,
                overlap=10,
                collection_name="test_collection"
            )
        self.addCleanup(self.chunk_system.client.delete_collection, "test_collection")
        
        self.test_text = (
            "Москва — столица России. В Москве много достопримечательностей. "
            "Санкт-Петербург — культурная столица. Челябинск — город металлургов."
        )
        
        # Добавляем документ
        self.chunk_system.add_document(self.test_text, doc_id="test_doc")

    def test_add_document(self):
        # Проверяем, что чанки создались и добавились в список
        chunks = self.chunk_system.get_chunks()
        self.assertGreater(len(chunks), 0)
        self.assertIn("test_doc", chunks[0]["id"])

    def test_search(self):
        # Проверяем поиск
        results = self.chunk_system.search("Москва", top_k=1)
        self.assertEqual(len(results), 1)
        
        # Проверяем структуру ответа: (dict, score)
        chunk_data, score = results[0]
        self.assertIn("Москва", chunk_data["text"])
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_clear(self):
        self.chunk_system.clear()
        self.assertEqual(len(self.chunk_system.get_chunks()), 0)
        # Проверяем, что поиск в пустой коллекции не падает
        results = self.chunk_system.search("любой запрос")
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
