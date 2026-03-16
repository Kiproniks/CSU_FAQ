import unittest
from EntityBased.EntityBased import EntityBased

class TestEntityBased(unittest.TestCase):
    
    def setUp(self):
        # Создаем экземпляр класса перед каждым тестом
        self.entity_system = EntityBased()
        
        # Добавляем тестовые данные
        self.test_chunks = [
            "Москва — столица России. В Москве много достопримечательностей",
            "Россия — великая страна с богатой историей",
            "Санкт-Петербург — культурная столица России",
            "Челябинск известен своими металлургическими предприятиями",
            "В России много природных ресурсов и красивых мест"
        ]
        
        for i, chunk in enumerate(self.test_chunks):
            self.entity_system.add_chunk(chunk, i)
            
        self.entity_system.build_index()

    def test_add_chunk(self):
        self.assertEqual(len(self.entity_system.chunks), len(self.test_chunks))
        
    def test_search(self):
        results = self.entity_system.search("столица России", top_k=2)
        self.assertGreater(len(results), 0)
        
    def test_clear(self):
        self.entity_system.clear()
        self.assertEqual(len(self.entity_system.chunks), 0)

if __name__ == '__main__':
    unittest.main()
