import matplotlib.pyplot as plt
from EntityBased.EntityBased import EntityBased
from ChunkBased.ChunkBased import ChunkBased
import textwrap

def visualize_search_results(entity_results, chunk_results, query: str, save_path="search_comparison.png"):
    """Красивая сравнительная визуализация поиска Entity vs Chunk"""
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Сравнение поиска по запросу:\n"{query}"', fontsize=16, fontweight='bold')

    # === EntityBased ===
    if entity_results:
        scores_e = [score for _, score in entity_results]
        axs[0].bar(range(len(scores_e)), scores_e, color='orange', alpha=0.8)
        axs[0].set_title('EntityBased (TF-IDF)')
        axs[0].set_ylabel('Score')
        axs[0].set_ylim(0, 1)
        axs[0].grid(axis='y', alpha=0.3)
        
        for i, score in enumerate(scores_e):
            axs[0].text(i, score + 0.02, f'{score:.3f}', ha='center')

    # === ChunkBased ===
    if chunk_results:
        scores_c = [score for _, score in chunk_results]
        axs[1].bar(range(len(scores_c)), scores_c, color='skyblue', alpha=0.8)
        axs[1].set_title('ChunkBased (Embeddings)')
        axs[1].set_ylabel('Score')
        axs[1].set_ylim(0, 1)
        axs[1].grid(axis='y', alpha=0.3)
        
        for i, score in enumerate(scores_c):
            axs[1].text(i, score + 0.02, f'{score:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Сравнительная визуализация сохранена: {save_path}\n")

    # Показываем лучшие чанки
    print("🔥 Лучший чанк из ChunkBased:")
    if chunk_results:
        best_chunk = chunk_results[0][0]['text']
        print(textwrap.fill(best_chunk[:600], width=100) + "...\n")


# ====================== Пример использования ======================
if __name__ == "__main__":
    print("Тест визуализации поиска...")
    # Здесь можно будет вызывать из основного теста