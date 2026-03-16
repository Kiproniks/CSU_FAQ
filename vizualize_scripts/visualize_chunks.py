import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from EntityBased.EntityBased import EntityBased
from ChunkBased.ChunkBased import ChunkBased


def visualize_entity_chunks(entity_system: EntityBased, title: str = "EntityBased - распределение сущностей"):
    """Визуализация для EntityBased: сколько сущностей в каждом чанке"""
    chunks = entity_system.get_chunks()
    if not chunks:
        print("Нет чанков в EntityBased")
        return
    
    entities_count = [len(chunk.get('entities', [])) if 'entities' in chunk else len(entity_system.chunk_entities[i]) 
                      for i, chunk in enumerate(chunks)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(entities_count)), entities_count, color='orange')
    plt.title(title)
    plt.xlabel('Номер чанка')
    plt.ylabel('Количество сущностей в чанке')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visual_entity_chunks.png')
    plt.show()
    print(f"✅ Визуализация EntityBased сохранена: visual_entity_chunks.png")


def visualize_chunk_based_chunks(chunk_system: ChunkBased, title: str = "ChunkBased - длины чанков"):
    """Визуализация для ChunkBased: длина каждого чанка и пересечения"""
    chunks = chunk_system.get_chunks()
    if not chunks:
        print("Нет чанков в ChunkBased")
        return
    
    lengths = [len(chunk['text']) for chunk in chunks]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(lengths)), lengths, color='skyblue')
    plt.axhline(y=chunk_system.chunk_size, color='red', linestyle='--', label=f'chunk_size = {chunk_system.chunk_size}')
    plt.title(title)
    plt.xlabel('Номер чанка')
    plt.ylabel('Длина чанка (символов)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visual_chunk_based_chunks.png')
    plt.show()
    print(f"✅ Визуализация ChunkBased сохранена: visual_chunk_based_chunks.png")


def visualize_both(entity_text: str, chunk_text: str = None):
    """Удобная функция — визуализирует сразу оба класса"""
    print("🔧 Создаём визуализацию для обоих подходов...\n")
    
    # EntityBased
    entity = EntityBased()
    # Разбиваем текст на чанки вручную (потому что в EntityBased нет авто-разбиения)
    chunk_size = 1000
    overlap = 200
    chunks_entity = [entity_text[i:i+chunk_size] for i in range(0, len(entity_text), chunk_size - overlap)]
    
    for i, ch in enumerate(chunks_entity):
        entity.add_chunk(ch, i)
    entity.build_index()
    
    visualize_entity_chunks(entity, "EntityBased — количество сущностей по чанкам")
    
    # ChunkBased
    chunk_system = ChunkBased(chunk_size=1000, overlap=200)
    chunk_system.add_document(chunk_text or entity_text, doc_id="visual_test")
    
    visualize_chunk_based_chunks(chunk_system, "ChunkBased — длины чанков с overlap")
    
    print("\n🎉 Визуализации готовы! Проверь папку проекта — там два PNG-файла.")


# ==================== Пример использования ====================
if __name__ == "__main__":
    # Пример текста для теста
    test_text = (
        "Гарри Поттер жил в доме номер четыре по улице Привит-драйв. "
        "Он был очень несчастлив, потому что его тетя и дядя ненавидели магию. "
        "В Хогвартсе Гарри учился вместе с Роном и Гермионой. "
        "Они сражались против Волан-де-Морта."
    )
    
    visualize_both(test_text)