"""Dependency-free Top-K evaluation helpers for TutorFlow RecSys.

These functions do not run Ollama or train models. They operate on saved
recommendation IDs and held-out relevant problem IDs.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence, Iterable
from typing import Hashable


def evaluate_top_k(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    relevant: Mapping[Hashable, Iterable[Hashable]],
    k: int = 10,
    *,
    seen: Mapping[Hashable, Iterable[Hashable]] | None = None,
) -> dict:
    """Macro-average HitRate/Recall/Precision@k across evaluable users.

    Users without relevant held-out items are excluded. For fair new-item
    recommendation evaluation, already-seen items must be filtered *before*
    forming Top-K; pass `seen` to verify this and remove any such items.
    Precision uses the number of returned items (up to k) as denominator;
    for a fixed-k ranking compare models with the same catalog/candidates.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    hit_sum = recall_sum = precision_sum = 0.0
    evaluated = 0
    details = []
    for user, relevant_items in relevant.items():
        targets = set(relevant_items)
        if not targets:
            continue
        already = set(seen.get(user, ())) if seen else set()
        # Relevant items present in training cannot be valid new-item targets.
        targets -= already
        if not targets:
            continue
        ranked = []
        unique = set()
        for item in recommendations.get(user, ()):
            if item in already or item in unique:
                continue
            unique.add(item)
            ranked.append(item)
            if len(ranked) == k:
                break
        n_hits = len(targets.intersection(ranked))
        hit = float(n_hits > 0)
        recall = n_hits / len(targets)
        precision = n_hits / len(ranked) if ranked else 0.0
        hit_sum += hit
        recall_sum += recall
        precision_sum += precision
        evaluated += 1
        details.append({"user": str(user), "relevant_count": len(targets),
                        "returned_count": len(ranked), "hits": n_hits,
                        "hit_rate": hit, "recall": recall, "precision": precision})
    return {
        "k": k,
        "users_evaluated": evaluated,
        "hit_rate_at_k": hit_sum / evaluated if evaluated else None,
        "recall_at_k": recall_sum / evaluated if evaluated else None,
        "precision_at_k": precision_sum / evaluated if evaluated else None,
        "per_user": details,
    }
