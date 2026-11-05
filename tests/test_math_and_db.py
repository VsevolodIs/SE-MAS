import sympy as sp
from src.database.db import init_db, add_student, get_student

def test_sympy_solver():
    x = sp.Symbol('x')
    expr = 2 * x + 4 - 10
    roots = sp.solve(expr, x)
    assert len(roots) == 1
    assert float(roots[0]) == 3.0

def test_db_init_and_student():
    init_db()
    student_id = add_student("Тест", grade=7, level="базовый", weak_topics="Уравнения")
    student = get_student("Тест")
    assert student is not None
    assert student["name"] == "Тест"
    assert student["grade"] == 7