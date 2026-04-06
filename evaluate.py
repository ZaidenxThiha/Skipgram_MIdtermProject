from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from scipy.stats import spearmanr

from skipgram import CORPUS, build_vocab, run_baseline, tokenize_corpus


WORD_PAIRS = [
    ("cat", "dog"),
    ("cat", "mat"),
    ("cat", "road"),
    ("dog", "road"),
    ("city", "road"),
    ("friends", "walk"),
]

QUERY_WORDS = ["cat", "dog", "city", "friends"]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns a scalar in [-1, 1].
    """
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def top_k_neighbors(
    word: str,
    embeddings: np.ndarray,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    k: int = 3,
) -> list[tuple[str, float]]:
    query_idx = word2idx[word]
    query_vec = embeddings[query_idx]
    neighbors: list[tuple[str, float]] = []

    for idx, candidate in idx2word.items():
        if idx == query_idx:
            continue
        score = cosine_similarity(query_vec, embeddings[idx])
        neighbors.append((candidate, score))

    neighbors.sort(key=lambda item: item[1], reverse=True)
    return neighbors[:k]


def pair_interpretation(word_a: str, word_b: str, score: float) -> str:
    if score >= 0.50:
        strength = "strong"
    elif score >= 0.20:
        strength = "moderate"
    elif score >= 0.00:
        strength = "weak"
    else:
        strength = "negative"

    if {word_a, word_b} == {"cat", "dog"}:
        meaning = "both are common household animals in the corpus"
    elif {word_a, word_b} == {"cat", "mat"}:
        meaning = "they co-occur directly in 'the cat sat on the mat'"
    elif {word_a, word_b} == {"cat", "road"}:
        meaning = "they never appear together, so similarity should stay low"
    elif {word_a, word_b} == {"dog", "road"}:
        meaning = "they appear together in the road sentence"
    elif {word_a, word_b} == {"city", "road"}:
        meaning = "they are linked by 'the road leads to the city'"
    elif {word_a, word_b} == {"friends", "walk"}:
        meaning = "they are semantically linked by the last sentence"
    else:
        meaning = "their relation is induced from local contexts"

    return f"{strength} similarity; {meaning}"


def neighbor_observation(word: str, neighbors: list[tuple[str, float]]) -> str:
    neighbor_words = [candidate for candidate, _ in neighbors]
    joined = ", ".join(neighbor_words)

    if word == "cat":
        return (
            f"Nearest words are {joined}. "
            "This mix shows one clear contextual match ('sat') plus some noisy neighbors, which is expected on a tiny corpus."
        )
    if word == "dog":
        return (
            f"Nearest words are {joined}. "
            "These are driven by shared local windows around dog, including subject-like and action-related contexts."
        )
    if word == "city":
        return (
            f"Nearest words are {joined}. "
            "Road-related neighbors are expected because city appears only once and is tied closely to 'road' and 'leads'. "
            "With such a tiny corpus, rare words often inherit narrow local structure."
        )
    if word == "friends":
        return (
            f"Nearest words are {joined}. "
            "Neighbors tend to reflect the final sentence and the earlier 'cat and the dog are friends' sentence. "
            "Words linked to social or motion contexts are reasonable here."
        )
    return f"Top neighbors for {word} are {joined}."


def train_gensim_model(tokenized_corpus: list[list[str]]) -> Word2Vec:
    return Word2Vec(
        sentences=tokenized_corpus,
        vector_size=10,
        window=2,
        sg=1,
        min_count=1,
        workers=1,
        seed=0,
        epochs=100,
    )


def format_neighbors(neighbors: list[tuple[str, float]]) -> str:
    return ", ".join(f"{word} ({score:.4f})" for word, score in neighbors)


def print_task_41(similarities: list[dict[str, object]]) -> None:
    print("Task 4.1")
    print("Cosine similarities using trained W_in embeddings (sorted)")
    print(f"{'Word Pair':<18} {'Cosine':>10}  Interpretation")
    for item in similarities:
        pair = item["pair"]
        print(f"{str(pair):<18} {item['score']:>10.4f}  {item['interpretation']}")
    print()


def print_task_42(neighbor_rows: list[dict[str, object]]) -> None:
    print("Task 4.2")
    print("Top-3 nearest neighbors from trained W_in embeddings")
    print(f"{'Query':<10} {'Top-3 Neighbors':<45} Linguistic Observation")
    for item in neighbor_rows:
        print(f"{item['query']:<10} {item['neighbors_text']:<45} {item['observation']}")
    print()


def print_task_43(
    comparison_rows: list[dict[str, object]],
    rho: float,
    p_value: float,
) -> None:
    print("Task 4.3")
    print("Extended similarity comparison with Gensim and rank positions")
    print(f"{'Word Pair':<18} {'Custom':>10} {'Gensim':>10} {'C-Rank':>8} {'G-Rank':>8}")
    for item in comparison_rows:
        print(
            f"{str(item['pair']):<18} {item['custom']:>10.4f} {item['gensim']:>10.4f} "
            f"{item['custom_rank']:>8} {item['gensim_rank']:>8}"
        )
    print()
    print(f"Spearman rho = {rho:.4f}")
    print(f"p-value = {p_value:.4f}")
    print()

    print("Technical differences")
    print("1. The custom model uses full-softmax cross-entropy, while Gensim Skip-gram uses negative sampling by default.")
    print("2. Gensim includes optimized training internals and a different update procedure, so embeddings will not numerically match even with the same corpus and seed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Part 4 for the Skip-gram project.")
    parser.add_argument("--output-dir", default=".", help="Directory for generated artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_baseline(output_dir)
    tokenized_corpus = tokenize_corpus(CORPUS)
    _, word2idx, idx2word = build_vocab(tokenized_corpus)
    custom_model = baseline["baseline_model"]
    custom_embeddings = custom_model.W_in

    similarity_rows: list[dict[str, object]] = []
    for word_a, word_b in WORD_PAIRS:
        score = cosine_similarity(custom_embeddings[word2idx[word_a]], custom_embeddings[word2idx[word_b]])
        similarity_rows.append(
            {
                "pair": (word_a, word_b),
                "score": score,
                "interpretation": pair_interpretation(word_a, word_b, score),
            }
        )
    similarity_rows.sort(key=lambda item: item["score"], reverse=True)

    neighbor_rows: list[dict[str, object]] = []
    for query in QUERY_WORDS:
        neighbors = top_k_neighbors(query, custom_embeddings, word2idx, idx2word, k=3)
        neighbor_rows.append(
            {
                "query": query,
                "neighbors": neighbors,
                "neighbors_text": format_neighbors(neighbors),
                "observation": neighbor_observation(query, neighbors),
            }
        )

    gensim_model = train_gensim_model(tokenized_corpus)
    comparison_rows: list[dict[str, object]] = []
    custom_scores: list[float] = []
    gensim_scores: list[float] = []

    for word_a, word_b in WORD_PAIRS:
        custom_score = cosine_similarity(custom_embeddings[word2idx[word_a]], custom_embeddings[word2idx[word_b]])
        gensim_score = float(gensim_model.wv.similarity(word_a, word_b))
        custom_scores.append(custom_score)
        gensim_scores.append(gensim_score)
        comparison_rows.append(
            {
                "pair": (word_a, word_b),
                "custom": custom_score,
                "gensim": gensim_score,
            }
        )

    rho, p_value = spearmanr(custom_scores, gensim_scores)
    custom_ranked = sorted(comparison_rows, key=lambda item: item["custom"], reverse=True)
    gensim_ranked = sorted(comparison_rows, key=lambda item: item["gensim"], reverse=True)
    custom_ranks = {item["pair"]: rank for rank, item in enumerate(custom_ranked, start=1)}
    gensim_ranks = {item["pair"]: rank for rank, item in enumerate(gensim_ranked, start=1)}

    for item in comparison_rows:
        item["custom_rank"] = custom_ranks[item["pair"]]
        item["gensim_rank"] = gensim_ranks[item["pair"]]

    print_task_41(similarity_rows)
    print_task_42(neighbor_rows)
    print_task_43(comparison_rows, float(rho), float(p_value))


if __name__ == "__main__":
    main()
