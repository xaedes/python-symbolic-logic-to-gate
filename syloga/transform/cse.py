
from syloga.ast.core import Assignment
from syloga.ast.containers import Tuple
from syloga.core.unused_symbols import unused_symbols
from syloga.transform.sympy_bridge import expr_to_sympy
from syloga.transform.sympy_bridge import sympy_to_expr

import sympy

def cse(expression, symbols=None):
    symbols = unused_symbols(expression,symbols)
    symbols = map(expr_to_sympy, symbols)
    return sympy_to_expr(sympy.cse(expr_to_sympy(expression),symbols=symbols))

def cse_assignments(assignments, *args, **kwargs):
    new_assignment_tuples, replaced_assignments = cse(assignments, *args, **kwargs)
    new_assignments = (Assignment(lhs,rhs) for lhs,rhs in new_assignment_tuples)
    return Tuple(*new_assignments, *replaced_assignments[0])
