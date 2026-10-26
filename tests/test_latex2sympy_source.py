"""Quick CPU regression checks for the vendored mathematical LaTeX parser.

Run after `uv sync`: `uv run --no-sync python -m unittest discover -s tests -v`.
"""

import unittest
from importlib.metadata import version


class BundledLatex2SympyTest(unittest.TestCase):
    def test_antlr_runtime_matches_bundled_parser(self):
        self.assertEqual(version("latex2sympy2"), "1.9.0")
        self.assertEqual(version("antlr4-python3-runtime"), "4.11.1")

    def test_bundled_and_sympy_latex_parsers(self):
        from sympy import Rational, simplify, sqrt
        from sympy.parsing.latex import parse_latex
        from latex2sympy2 import latex2sympy

        cases = (
            (r"\frac{1}{2}", Rational(1, 2)),
            (r"\sqrt{2}", sqrt(2)),
            (r"x^2 + 2x + 1", None),
        )
        for formula, expected in cases:
            with self.subTest(formula=formula):
                local = latex2sympy(formula)
                sympy_builtin = parse_latex(formula)
                # The bundled parser preserves unevaluated SymPy nodes.
                # Structural == can be False for mathematically equal expressions
                # (e.g. Pow(2, -1, evaluate=False) versus Rational(1, 2)).
                if expected is not None:
                    self.assertEqual(simplify(local - expected), 0)
                    self.assertEqual(simplify(sympy_builtin - expected), 0)
                self.assertEqual(simplify(local - sympy_builtin), 0)

    def test_structural_vs_mathematical_equality(self):
        from sympy import Integer, Pow, Rational, simplify

        unevaluated = Pow(Integer(2), Integer(-1), evaluate=False)
        expected = Rational(1, 2)
        self.assertNotEqual(unevaluated, expected)
        self.assertEqual(simplify(unevaluated - expected), 0)


if __name__ == "__main__":
    unittest.main()
