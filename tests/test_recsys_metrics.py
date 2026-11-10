import unittest
from src.models.recsys_metrics import evaluate_top_k


class MetricsTest(unittest.TestCase):
    def test_hit_rate_is_not_recall(self):
        result = evaluate_top_k({"student": [1, 9, 8, 7]}, {"student": [1, 2, 3, 4]}, k=4)
        self.assertEqual(result["hit_rate_at_k"], 1.0)
        self.assertEqual(result["recall_at_k"], 0.25)
        self.assertEqual(result["precision_at_k"], 0.25)

    def test_seen_items_excluded(self):
        result = evaluate_top_k({"student": [1, 2, 3]}, {"student": [1, 2]}, k=2, seen={"student": [1]})
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["hit_rate_at_k"], 1.0)

    def test_empty_targets_excluded(self):
        result = evaluate_top_k({"a": [1]}, {"a": [], "b": [1]}, k=1)
        self.assertEqual(result["users_evaluated"], 1)
        self.assertEqual(result["hit_rate_at_k"], 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            evaluate_top_k({}, {}, k=0)


if __name__ == "__main__":
    unittest.main()
