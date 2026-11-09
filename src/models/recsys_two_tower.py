from scripts.recsys_metrics import evaluate_top_k
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MathRecDataset(Dataset):

    def __init__(self, df, total_problems, neg_ratio=4):
        self.students = df["student_idx"].values
        self.problems = df["problem_idx"].values
        self.total_problems = total_problems
        self.neg_ratio = neg_ratio

        self.positive_by_student = (
            df.groupby("student_idx")["problem_idx"]
            .apply(set)
            .to_dict()
        )

        self.negative_by_student = {}

        for student_id, positives in self.positive_by_student.items():
            candidates = [
                p for p in range(total_problems)
                if p not in positives
            ]

            if not candidates and neg_ratio > 0:
                raise ValueError(
                    f"Нет отрицательных примеров для ученика {student_id}"
                )

            self.negative_by_student[student_id] = candidates

    def __len__(self):
        return len(self.students) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.neg_ratio)
        student_id = int(self.students[pos_idx])

        if idx % (1 + self.neg_ratio) == 0:
            problem_id = int(self.problems[pos_idx])
            label = 1.0
        else:
            problem_id = int(
                np.random.choice(
                    self.negative_by_student[student_id]
                )
            )
            label = 0.0

        return (
            torch.tensor(student_id, dtype=torch.long),
            torch.tensor(problem_id, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )

class TutorFlowTwoTower(nn.Module):

    def __init__(self, num_students, num_problems, emb_dim=32):
        super().__init__()
        self.student_emb = nn.Embedding(num_students, emb_dim)
        self.student_mlp = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, emb_dim),
        )

        self.problem_emb = nn.Embedding(num_problems, emb_dim)
        self.problem_mlp = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, emb_dim),
        )

    def forward(self, student_ids, problem_ids):
        s_vec = self.student_mlp(self.student_emb(student_ids))
        p_vec = self.problem_mlp(self.problem_emb(problem_ids))

        s_norm = nn.functional.normalize(s_vec, p=2, dim=1)
        p_norm = nn.functional.normalize(p_vec, p=2, dim=1)

        return torch.sum(s_norm * p_norm, dim=1)


def generate_math_interactions():
    np.random.seed(42)
    num_students = 150
    num_problems = 80

    topics = [
        "linear_equations",
        "fractions",
        "percentages",
        "quadratic_basics",
        "word_problems",
    ]
    records = []

    for s_id in range(num_students):
        fav_topics = np.random.choice(topics, size=2, replace=False)
        n_solved = np.random.randint(6, 15)

        for _ in range(n_solved):
            p_id = np.random.randint(0, num_problems)
            records.append({
                "student_idx": s_id,
                "problem_idx": p_id,
                "timestamp": np.random.randint(1700000000, 1710000000),
            })

    df = pd.DataFrame(records).sort_values(by=["student_idx", "timestamp"])

    train_rows, test_rows = [], []
    for _, grp in df.groupby("student_idx"):
        n = len(grp)
        split_point = int(np.floor(0.8 * n))
        train_rows.append(grp.iloc[:split_point])
        test_rows.append(grp.iloc[split_point:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df, num_students, num_problems


if __name__ == "__main__":
    train_df, test_df, n_students, n_problems = generate_math_interactions()
    print(
        f"TutorFlow RecSys: {n_students} учеников, {n_problems} задач в базе"
    )
    print(f"Обучающая выборка: {len(train_df)}, тестовая: {len(test_df)}")

    torch.manual_seed(42)
    model = TutorFlowTwoTower(n_students, n_problems, emb_dim=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

    train_loader = DataLoader(
        MathRecDataset(train_df, n_problems, neg_ratio=4),
        batch_size=64,
        shuffle=True,
    )

    print("\nОбучение нейросети Two-Tower подбора задач")
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for s_b, p_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(s_b, p_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Эпоха {epoch + 1}/5 | Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        all_p_ids = torch.arange(n_problems, dtype=torch.long)
        all_p_vecs = model.problem_mlp(model.problem_emb(all_p_ids))
        all_p_vecs = nn.functional.normalize(all_p_vecs, p=2, dim=1)

        hits, total = 0, 0
        ground_truth_map = (
            test_df.groupby("student_idx")["problem_idx"]
            .apply(set)
            .to_dict()
        )

        seen_map = (
            train_df.groupby("student_idx")["problem_idx"]
            .apply(set)
            .to_dict()
        )

        recommendations = {}

        for s_idx in ground_truth_map:
            s_t = torch.tensor([s_idx], dtype=torch.long)

            s_vec = model.student_mlp(model.student_emb(s_t))
            s_vec = nn.functional.normalize(s_vec, p=2, dim=1)

            scores = torch.matmul(s_vec, all_p_vecs.T).squeeze(0)

            # Сортируем все задания по релевантности
            ranked = torch.argsort(scores, descending=True).tolist()

            # Убираем уже просмотренные задания
            seen = seen_map.get(s_idx, set())
            recommendations[s_idx] = [
                p for p in ranked if p not in seen
            ][:10]

        metrics = evaluate_top_k(
            recommendations=recommendations,
            relevant=ground_truth_map,
            seen=seen_map,
            k=10,
        )

    print("\nРезультаты TutorFlow RecSys (Two-Tower)")
    print(f"Оценено учеников: {metrics['users_evaluated']}")
    print(f"HitRate@10: {metrics['hit_rate_at_k']:.4f}")
    print(f"Recall@10: {metrics['recall_at_k']:.4f}")
    print(f"Precision@10: {metrics['precision_at_k']:.4f}")