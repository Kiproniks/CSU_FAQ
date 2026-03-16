import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import textwrap
from EntityBased.EntityBased import EntityBased
from ChunkBased.ChunkBased import ChunkBased

def visualize_search_comparison(entity_results, chunk_results, query: str, save_path="search_comparison.png"):
    """
    Красивая визуализация сравнения двух подходов + подсветка лучших чанков
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Сравнение поиска RAG по запросу:\n"{query}"', fontsize=16, fontweight='bold')

    # EntityBased
    if entity_results:
        scores_e = [score for _, score in entity_results]
        labels_e = [f"Чанк {r[0]['id']}" for r in entity_results]
        ax1.bar(range(len(scores_e)), scores_e, color='orange', alpha=0.85)
        ax1.set_title('EntityBased (TF-IDF + ключевые слова)')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(scores_e)))
        ax1.set_xticklabels(labels_e, rotation=45)
        for i, s in enumerate(scores_e):
            ax1.text(i, s + 0.02, f'{s:.3f}', ha='center', fontsize=10)

    # ChunkBased
    if chunk_results:
        scores_c = [score for _, score in chunk_results]
        labels_c = [f"Чанк {r[0]['metadata'].get('chunk_index', i)}" for i, r in enumerate(chunk_results)]
        ax2.bar(range(len(scores_c)), scores_c, color='skyblue', alpha=0.85)
        ax2.set_title('ChunkBased (Embeddings + семантика)')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(len(scores_c)))
        ax2.set_xticklabels(labels_c, rotation=45)
        for i, s in enumerate(scores_c):
            ax2.text(i, s + 0.02, f'{s:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"✅ Сравнительная визуализация сохранена как: {save_path}\n")

    # Показ лучших чанков
    print("🔥 ТОП-1 ChunkBased (самый релевантный):")
    if chunk_results:
        best = chunk_results[0][0]['text']
        print(textwrap.fill(best[:700], width=110))
        print("\n" + "="*80 + "\n")


# ====================== Тест ======================
if __name__ == "__main__":
    print("Запуск тестовой визуализации...")
    # Можно будет вызывать из test_harry_potter.py