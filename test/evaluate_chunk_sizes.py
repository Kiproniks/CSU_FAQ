import os
import sys
import time
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ollama
from pypdf import PdfReader

# Чтобы импортировать ChunkBasedTest.py из этой же папки
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ChunkBasedTest import ChunkBased
from test_questions import TEST_FAQ


# ================== НАСТРОЙКИ ==================
CHUNK_SIZES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
NUM_QUESTIONS = 30
BOOKS_FOLDER = "../harry_potter"
LLM_MODEL = "llama3.2:3b"
TOP_K = 4


# ================== ЗАГРУЗКА PDF ==================
def load_all_books() -> str:
    if not os.path.exists(BOOKS_FOLDER):
        raise FileNotFoundError(f"Папка {BOOKS_FOLDER} не найдена!")

    all_text = ""
    pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.lower().endswith(".pdf")]
    print(f"✅ Найдено PDF книг: {len(pdf_files)}")

    for filename in sorted(pdf_files):
        path = os.path.join(BOOKS_FOLDER, filename)
        reader = PdfReader(path)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"

        print(f"   Обработан: {filename}")

    return all_text.strip()


# ================== ВОПРОСЫ ==================
questions = TEST_FAQ[:NUM_QUESTIONS]
print(f"✅ Загружено {len(questions)} вопросов для теста\n")

all_text = load_all_books()

# ================== ВРЕМЕННАЯ БАЗА ДЛЯ ТЕСТА ==================
test_db_path = os.path.join(os.path.dirname(__file__), "chroma_db_test")
shutil.rmtree(test_db_path, ignore_errors=True)

results = []

try:
    for chunk_size in CHUNK_SIZES:
        print(f"🔬 Тестируем chunk_size = {chunk_size} ...")

        collection_name = f"test_chunk_{chunk_size}"

        retriever = ChunkBased(
            chunk_size=chunk_size,
            overlap=max(50, chunk_size // 6),
            collection_name=collection_name,
        )

        # На всякий случай чистим коллекцию, если вдруг осталась с прошлого запуска
        try:
            retriever.client.delete_collection(collection_name)
        except Exception:
            pass

        # Создаём новый retriever после очистки
        retriever = ChunkBased(
            chunk_size=chunk_size,
            overlap=max(50, chunk_size // 6),
            collection_name=collection_name,
        )

        # --- Индексация ---
        index_start = time.perf_counter()
        retriever.add_document(all_text, doc_id="all_books", metadata={"source": "hp"})
        index_time = time.perf_counter() - index_start

        # --- Тест вопросов ---
        query_times = []
        scores = []

        for item in questions:
            q = item["question"]
            ideal = item["ideal"]

            q_start = time.perf_counter()

            results_search = retriever.search(q, top_k=TOP_K)
            context = "\n\n".join([r[0]["text"] for r in results_search])

            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": f"Контекст:\n{context}\n\nВопрос: {q}\nОтветь точно и кратко:"
                    }
                ]
            )
            answer = resp["message"]["content"].strip()

            elapsed = time.perf_counter() - q_start
            query_times.append(elapsed)

            judge_prompt = (
                f"Вопрос: {q}\n"
                f"Идеальный: {ideal}\n"
                f"Ответ бота: {answer}\n"
                f"Оцени 0-100 (только число):"
            )

            try:
                judge = ollama.chat(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": judge_prompt}]
                )
                score = int(judge["message"]["content"].strip())
            except Exception:
                score = 50

            scores.append(score)

        avg_query = float(np.mean(query_times))
        avg_score = float(np.mean(scores))

        results.append({
            "chunk_size": chunk_size,
            "index_time_sec": round(index_time, 2),
            "avg_query_time_sec": round(avg_query, 2),
            "avg_quality_percent": round(avg_score, 1)
        })

        print(
            f"   Индексация: {index_time:.1f}с | "
            f"Ответ: {avg_query:.2f}с | "
            f"Качество: {avg_score:.1f}%\n"
        )

        # Удаляем коллекцию после каждого размера чанка
        try:
            retriever.client.delete_collection(collection_name)
        except Exception:
            pass

finally:
    # Удаляем всю тестовую базу после завершения теста
    shutil.rmtree(test_db_path, ignore_errors=True)


# ================== ГРАФИКИ ==================
df = pd.DataFrame(results)
df.to_csv("chunk_results.csv", index=False, encoding="utf-8")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
sizes = df["chunk_size"]

ax1.plot(sizes, df["index_time_sec"], "o-", linewidth=3, label="Время индексации")
ax1.plot(sizes, df["avg_query_time_sec"], "s-", linewidth=3, label="Среднее время ответа")
ax1.set_title("Время vs Размер чанка")
ax1.set_ylabel("Секунды")
ax1.legend()
ax1.grid(True)

ax2.plot(sizes, df["avg_quality_percent"], "^--", linewidth=3, label="Качество (%)")
ax2.set_title("Эффективность vs Размер чанка")
ax2.set_xlabel("Chunk Size")
ax2.set_ylabel("Качество ответа (%)")
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("chunk_size_evaluation.png", dpi=300)
plt.show()

print("🎉 Готово! График сохранён: chunk_size_evaluation.png")