from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


CORPUS = [
    "the cat sat on the mat",
    "the dog ran on the road",
    "the cat and the dog are friends",
    "she loves her cat",
    "he walks his dog every day",
    "the mat is on the floor",
    "the road leads to the city",
    "friends walk together every day",
]


def tokenize_corpus(corpus: Iterable[str]) -> list[list[str]]:
    return [sentence.lower().split() for sentence in corpus]


def build_vocab(tokenized_corpus: Iterable[Iterable[str]]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocab = sorted({token for sentence in tokenized_corpus for token in sentence})
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word


def generate_pairs(tokenized_corpus: Iterable[Iterable[str]], word2idx: dict[str, int], window_size: int = 2) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        for center_pos, center_idx in enumerate(indices):
            left = max(0, center_pos - window_size)
            right = min(len(indices), center_pos + window_size + 1)
            for context_pos in range(left, right):
                if context_pos == center_pos:
                    continue
                pairs.append((center_idx, indices[context_pos]))
    return pairs


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def cross_entropy_loss(y_hat: np.ndarray, target_idx: int) -> float:
    return float(-np.log(y_hat[target_idx] + 1e-12))


@dataclass
class GradientCheckResult:
    matrix: str
    index: tuple[int, int]
    analytical: float
    numerical: float
    relative_error: float


class SkipGram:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0, init_scale: float = 0.01):
        np.random.seed(seed)
        self.V = vocab_size
        self.d = embed_dim
        self.W_in = np.random.randn(vocab_size, embed_dim) * init_scale
        self.W_out = np.random.randn(embed_dim, vocab_size) * init_scale

    def forward(self, center_idx: int) -> tuple[np.ndarray, np.ndarray]:
        v_c = self.W_in[center_idx].copy()
        scores = self.W_out.T @ v_c
        y_hat = softmax(scores)
        return v_c, y_hat

    def backward(
        self,
        center_idx: int,
        context_idx: int,
        v_c: np.ndarray,
        y_hat: np.ndarray,
        lr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        error = y_hat.copy()
        error[context_idx] -= 1.0

        grad_W_out = np.outer(v_c, error)
        grad_v_c = self.W_out @ error

        self.W_out -= lr * grad_W_out
        self.W_in[center_idx] -= lr * grad_v_c

        return error, grad_W_out, grad_v_c, self.W_in[center_idx].copy()

    def loss_for_pair(self, center_idx: int, context_idx: int) -> float:
        _, y_hat = self.forward(center_idx)
        return cross_entropy_loss(y_hat, context_idx)

    def analytical_gradients(self, center_idx: int, context_idx: int) -> tuple[np.ndarray, np.ndarray]:
        v_c, y_hat = self.forward(center_idx)
        error = y_hat.copy()
        error[context_idx] -= 1.0
        grad_W_out = np.outer(v_c, error)
        grad_W_in = np.zeros_like(self.W_in)
        grad_W_in[center_idx] = self.W_out @ error
        return grad_W_in, grad_W_out


def numerical_gradient_for_entry(
    model: SkipGram,
    matrix_name: str,
    index: tuple[int, int],
    center_idx: int,
    context_idx: int,
    eps: float,
) -> float:
    matrix = getattr(model, matrix_name)
    original_value = matrix[index]

    matrix[index] = original_value + eps
    loss_plus = model.loss_for_pair(center_idx, context_idx)

    matrix[index] = original_value - eps
    loss_minus = model.loss_for_pair(center_idx, context_idx)

    matrix[index] = original_value
    return float((loss_plus - loss_minus) / (2 * eps))


def test_gradients(
    model: SkipGram,
    center_idx: int,
    context_idx: int,
    eps: float = 1e-5,
    num_checks: int = 5,
    seed: int = 123,
) -> tuple[bool, list[GradientCheckResult]]:
    rng = np.random.RandomState(seed)
    grad_W_in, grad_W_out = model.analytical_gradients(center_idx, context_idx)
    results: list[GradientCheckResult] = []

    win_columns = rng.choice(model.d, size=min(num_checks, model.d), replace=False)
    for col in win_columns:
        index = (center_idx, int(col))
        analytical = float(grad_W_in[index])
        numerical = numerical_gradient_for_entry(model, "W_in", index, center_idx, context_idx, eps)
        relative_error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-12)
        results.append(
            GradientCheckResult(
                matrix="W_in",
                index=index,
                analytical=analytical,
                numerical=numerical,
                relative_error=relative_error,
            )
        )

    checked = set()
    while len(checked) < num_checks:
        index = tuple(int(rng.randint(dim)) for dim in grad_W_out.shape)
        if index in checked:
            continue
        checked.add(index)
        analytical = float(grad_W_out[index])
        numerical = numerical_gradient_for_entry(model, "W_out", index, center_idx, context_idx, eps)
        relative_error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-12)
        results.append(
            GradientCheckResult(
                matrix="W_out",
                index=index,
                analytical=analytical,
                numerical=numerical,
                relative_error=relative_error,
            )
        )

    return all(result.relative_error < 1e-5 for result in results), results


def train(
    model: SkipGram,
    pairs: list[tuple[int, int]],
    epochs: int,
    lr_init: float,
    lr_decay: float = 0.005,
    shuffle_seed: int = 0,
) -> list[float]:
    losses: list[float] = []
    rng = np.random.RandomState(shuffle_seed)

    for epoch in range(1, epochs + 1):
        lr = lr_init / (1.0 + lr_decay * epoch)
        shuffled_indices = rng.permutation(len(pairs))
        total_loss = 0.0
        for pair_idx in shuffled_indices:
            center_idx, context_idx = pairs[pair_idx]
            v_c, y_hat = model.forward(center_idx)
            total_loss += cross_entropy_loss(y_hat, context_idx)
            model.backward(center_idx, context_idx, v_c, y_hat, lr)
        losses.append(total_loss / len(pairs))
    return losses


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def top_k_neighbors(model: SkipGram, word: str, word2idx: dict[str, int], idx2word: dict[int, str], k: int = 3) -> list[tuple[str, float]]:
    query_idx = word2idx[word]
    query_vec = model.W_in[query_idx]
    scores: list[tuple[str, float]] = []
    for idx, candidate in idx2word.items():
        if idx == query_idx:
            continue
        scores.append((candidate, cosine_similarity(query_vec, model.W_in[idx])))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:k]


def plot_losses(losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 101), losses, color="navy", lw=1.5, label="Avg. loss")
    for epoch, color in ((10, "orange"), (50, "green"), (100, "red")):
        plt.axvline(
            epoch,
            linestyle="--",
            color=color,
            lw=1.2,
            label=f"Epoch {epoch} (loss={losses[epoch - 1]:.3f})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.title("Skip-gram Training Loss Curve (d=10, W=2, η₀=0.025)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_array(values: np.ndarray, precision: int = 4) -> str:
    return np.array2string(values, precision=precision, suppress_small=False)


def run_baseline(output_dir: Path) -> dict[str, object]:
    tokenized = tokenize_corpus(CORPUS)
    vocab, word2idx, idx2word = build_vocab(tokenized)
    pairs = generate_pairs(tokenized, word2idx, window_size=2)
    unique_pairs = sorted(set(pairs))

    verification_model = SkipGram(len(vocab), embed_dim=3, seed=42, init_scale=0.1)
    cat_idx = word2idx["cat"]
    sat_idx = word2idx["sat"]
    v_c_small, y_hat_small = verification_model.forward(cat_idx)
    sat_score = float(verification_model.W_out[:, sat_idx] @ v_c_small)
    sat_prob = float(y_hat_small[sat_idx])
    sat_loss = cross_entropy_loss(y_hat_small, sat_idx)
    error, grad_W_out, grad_v_c, updated_v_cat = verification_model.backward(cat_idx, sat_idx, v_c_small, y_hat_small, lr=0.1)

    forward_model = SkipGram(len(vocab), embed_dim=10, seed=0)
    the_idx = word2idx["the"]
    _, y_hat_the = forward_model.forward(the_idx)

    gradient_model = SkipGram(len(vocab), embed_dim=10, seed=0)
    gradient_ok, gradient_results = test_gradients(gradient_model, the_idx, word2idx["cat"])

    train_model = SkipGram(len(vocab), embed_dim=10, seed=0)
    losses = train(train_model, pairs, epochs=100, lr_init=0.025, lr_decay=0.005, shuffle_seed=0)
    plot_losses(losses, output_dir / "loss_curve.png")

    dim20_model = SkipGram(len(vocab), embed_dim=20, seed=0)
    losses_dim20 = train(dim20_model, pairs, epochs=100, lr_init=0.025, lr_decay=0.005, shuffle_seed=0)

    fixed_lr_model = SkipGram(len(vocab), embed_dim=10, seed=0)
    losses_fixed_lr = train(fixed_lr_model, pairs, epochs=100, lr_init=0.05, lr_decay=0.0, shuffle_seed=0)

    return {
        "tokenized": tokenized,
        "vocab": vocab,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "pairs": pairs,
        "unique_pairs": unique_pairs,
        "verification": {
            "v_cat": v_c_small,
            "sat_score": sat_score,
            "sat_prob": sat_prob,
            "sat_loss": sat_loss,
            "error": error,
            "grad_v_c": grad_v_c,
            "updated_v_cat": updated_v_cat,
            "updated_w_out_sat": verification_model.W_out[:, sat_idx].copy(),
            "grad_w_out_sat": grad_W_out[:, sat_idx].copy(),
        },
        "forward_check": {
            "sum": float(y_hat_the.sum()),
            "max": float(y_hat_the.max()),
        },
        "gradient_check": {
            "passed": gradient_ok,
            "results": gradient_results,
        },
        "baseline_model": train_model,
        "losses": losses,
        "experiments": {
            "dim20": losses_dim20,
            "fixed_lr": losses_fixed_lr,
        },
    }


def print_summary(results: dict[str, object]) -> None:
    vocab = results["vocab"]
    pairs = results["pairs"]
    unique_pairs = results["unique_pairs"]
    verification = results["verification"]
    forward_check = results["forward_check"]
    gradient_check = results["gradient_check"]
    losses = results["losses"]
    experiments = results["experiments"]
    word2idx = results["word2idx"]

    print("Task 1.4")
    print("Vocabulary:")
    print(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training pairs count: {len(pairs)}")
    print(f"Unique ordered pair types: {len(unique_pairs)}")
    print()

    print("Task 2.1")
    print("Verification with seed=42, d=3, pair=('cat','sat')")
    print(f"v_cat = {format_array(verification['v_cat'])}")
    print(f"score('sat') = {verification['sat_score']:.4f}")
    print(f"P(sat | cat) = {verification['sat_prob']:.4f}")
    print(f"loss = {verification['sat_loss']:.4f}")
    print()

    print("Task 2.2")
    print("Verification with seed=0, d=10, center='the'")
    print(f"y_hat.sum() = {forward_check['sum']:.10f}")
    print(f"max(y_hat) = {forward_check['max']:.10f}")
    print()

    print("Task 2.3")
    print("Manual backward verification")
    print(f"error[sat] = {verification['error'][word2idx['sat']]:.4f}")
    print(f"updated v_cat = {format_array(verification['updated_v_cat'])}")
    print(f"updated W_out[:, sat] = {format_array(verification['updated_w_out_sat'])}")
    print(f"grad_v_cat = {format_array(verification['grad_v_c'])}")
    print(f"grad_W_out[:, sat] = {format_array(verification['grad_w_out_sat'])}")
    print()

    print("Task 2.4")
    print(f"Gradient check passed = {gradient_check['passed']}")
    for result in gradient_check["results"]:
        print(
            f"{result.matrix}{result.index}: analytical={result.analytical:.8f}, "
            f"numerical={result.numerical:.8f}, rel_error={result.relative_error:.3e}"
        )
    print()

    print("Task 3.1")
    print("Training checkpoints")
    for epoch in (10, 50, 100):
        print(f"epoch {epoch}: {losses[epoch - 1]:.4f}")
    print()

    print("Task 3.3")
    print("Hyperparameter experiments")
    print(f"d=20 -> epoch50={experiments['dim20'][49]:.4f}, epoch100={experiments['dim20'][99]:.4f}")
    print(f"fixed lr=0.05 -> epoch50={experiments['fixed_lr'][49]:.4f}, epoch100={experiments['fixed_lr'][99]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify a Skip-gram model from scratch.")
    parser.add_argument("--output-dir", default=".", help="Directory for generated artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_baseline(output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
