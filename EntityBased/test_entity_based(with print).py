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
        print("\nТестовый набор данных загружен и проиндексирован")

    def test_add_chunk(self):
        print("\nПроверка добавления чанков:")
        print(f"Ожидаемое количество: {len(self.test_chunks)}")
        print(f"Фактическое количество: {len(self.entity_system.chunks)}")
        self.assertEqual(len(self.entity_system.chunks), len(self.test_chunks))
        
    def test_search(self):
        print("\nПроверка поиска по запросу 'столица России':")
        results = self.entity_system.search("столица России", top_k=2)
        print("Найденные результаты:")
        for idx, (chunk, score) in enumerate(results):
            print(f"Результат {idx+1}:")
            print(f"  Оценка релевантности: {score:.4f}")
            print(f"  Текст: {chunk['text']}")
        self.assertGreater(len(results), 0)
        
    def test_clear(self):
        print("\nПроверка очистки системы:")
        self.entity_system.clear()
        print(f"Количество чанков после очистки: {len(self.entity_system.chunks)}")
        self.assertEqual(len(self.entity_system.chunks), 0)

if __name__ == '__main__':
    print("\nЗапуск тестового набора...")
    unittest.main()
