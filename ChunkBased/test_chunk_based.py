import unittest
import os
import shutil
from ChunkBased import ChunkBased

class TestChunkBased(unittest.TestCase):
    
    def setUp(self):
        # Используем уникальное имя коллекции для тестов, чтобы не затереть основную базу
        self.db_path = "./test_chroma_db"
        self.chunk_system = ChunkBased(
            chunk_size=50, 
            overlap=10, 
            collection_name="test_collection"
        )
        
        self.test_text = (
            "Москва — столица России. В Москве много достопримечательностей. "
            "Санкт-Петербург — культурная столица. Челябинск — город металлургов."
        )
        
        # Добавляем документ
        self.chunk_system.add_document(self.test_text, doc_id="test_doc")

    def tearDown(self):
        # Очищаем ресурсы после теста
        self.chunk_system.clear()
        # Удаляем папку тестовой БД, если она создалась
        if os.path.exists("./chroma_db"):
             pass # Chroma сама управляет файлами, но в тестах лучше делать clear()

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