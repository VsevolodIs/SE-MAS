import unittest

import pandas as pd

from src.models.recsys_two_tower import MathRecDataset


class NegativeSamplingTest(unittest.TestCase):
    def test_negatives_exclude_train_positives(self):
        df = pd.DataFrame({
            "student_idx": [0, 0, 1],
            "problem_idx": [1, 3, 2],
        })

        dataset = MathRecDataset(
            df,
            total_problems=5,
            neg_ratio=4,
        )

        positives = {
            0: {1, 3},
            1: {2},
        }

        for idx in range(len(dataset)):
            student, problem, label = dataset[idx]

            if label.item() == 0.0:
                self.assertNotIn(
                    problem.item(),
                    positives[student.item()],
                )


if __name__ == "__main__":
    unittest.main()