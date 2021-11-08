
from syloga.ast.core import Expression
from syloga.core.truth_table import truth_table_by_repr

def assert_equality_by_table(expr, expression):
    if isinstance(expression, Expression):
        try:
            assert(truth_table_by_repr(expression) == truth_table_by_repr(expr))
        except:
            print(expression)
            print(expr)
            raise
    #print("return ", type(expr), expr)
    return expr
