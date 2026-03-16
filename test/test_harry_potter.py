import sys
import os

# Добавляем корневую папку проекта в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ПРАВИЛЬНЫЕ ИМПОРТЫ (учитывая твою структуру с папками)
from EntityBased.EntityBased import EntityBased
from ChunkBased.ChunkBased import ChunkBased
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text.strip()


# ====================== ОСНОВНОЙ ТЕСТ ======================
if __name__ == "__main__":
    # Правильный относительный путь (учитывая, что скрипт в папке test/)
    pdf_path = "../harry_potter/1_GP_i_FK_Rosmen.pdf"

    print("📖 Извлекаем текст из PDF...")
    
    # Дополнительная проверка пути
    if not os.path.exists(pdf_path):
        pdf_path = "harry_potter/1_GP_i_FK_Rosmen.pdf"  # если запустили из корня
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"Файл не найден!\n"
                f"Проверенные пути:\n"
                f"  - ../harry_potter/1_GP_i_FK_Rosmen.pdf\n"
                f"  - harry_potter/1_GP_i_FK_Rosmen.pdf"
            )

    full_text = extract_text_from_pdf(pdf_path)
    print(f"✅ Извлечено {len(full_text):,} символов\n")
    
    print("🔹 Тестируем EntityBased...")
    entity = EntityBased()
    chunk_size = 1000
    overlap = 200
    step = chunk_size - overlap

    for i in range(0, len(full_text), step):
        chunk = full_text[i:i + chunk_size]
        if len(chunk) > 100:   # отбрасываем совсем мелкие кусочки
            entity.add_chunk(chunk, i)
    entity.build_index()
    print(f"   Добавлено {len(entity.get_chunks())} чанков в EntityBased\n")
    
    print("🔹 Тестируем ChunkBased...")
    chunk_system = ChunkBased(chunk_size=1000, overlap=200)
    chunk_system.add_document(full_text, doc_id="Harry_Potter_1")
    print(f"   Добавлено {len(chunk_system.get_chunks())} чанков в ChunkBased\n")
    
    # Пример поиска
    query = "Как Гарри попал в Хогвартс?"
    print(f'🔍 Поиск по запросу: "{query}"\n')
    
    print("EntityBased результаты:")
    entity_results = entity.search(query, top_k=3)
    for i, (ch, score) in enumerate(entity_results):
        print(f"{i+1}. Score: {score:.3f} | {ch['text'][:200]}...\n")
    
    print("ChunkBased результаты:")
    chunk_results = chunk_system.search(query, top_k=3)
    for i, (ch, score) in enumerate(chunk_results):
        print(f"{i+1}. Score: {score:.3f} | {ch['text'][:200]}...\n")
    
    # Визуализация (исправленный импорт)
    print("\n📊 Создаём визуализацию чанков...")
    try:
        from vizualize_scripts.visualize_chunks import visualize_both
        visualize_both(full_text[:15000])   # первые 15к символов для скорости
    except ImportError:
        print("⚠️ Не удалось импортировать visualize_both. Проверьте путь к visualize_chunks.py")