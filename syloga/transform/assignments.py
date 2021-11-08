
import itertools


from syloga.ast.containers import Tuple
from syloga.ast.core import Assignment
from syloga.ast.core import BreakOut
from syloga.ast.core import Expression
from syloga.ast.traversal import iter_assignments
from syloga.ast.traversal import preorder_traversal
from syloga.ast.traversal import zip_expression
from syloga.core.unused_symbols import unused_symbols
from syloga.transform.basic import replace
from syloga.transform.core import map_expression_args
from syloga.transform.replace_assignments_by_lhs import replace_assignments_by_lhs
from syloga.utils.functional import identity
from syloga.utils.functional import iter_unique
from syloga.utils.predicates import is_mappable_collection




def flatten_assignments(expression):
    zip_assignment = lambda x: zip_expression(x.lhs, x.rhs, key=is_lvalue, flat=True, as_list=False)
    flat_assignment = lambda x: list(itertools.starmap(Assignment, zip_assignment(x)))
    wrap_tuple = lambda x: x if len(x) == 1 else Tuple(*x)
    return replace(
        expression,
        Assignment,
        lambda x,r: wrap_tuple(flat_assignment(map_expression_args(r,x)))
    )

def get_assignments(expression, traversal = preorder_traversal, only_unique=True, sort_key=str):
    assignments = iter_assignments(expression, traversal, only_unique, sort_key)
    return Tuple(*assignments)

def iter_assignments(expression, traversal = preorder_traversal, only_unique=True, sort_key=str):
    is_assignment = lambda x:type(x)==Assignment
    assignments = filter(is_assignment,traversal(expression))

    replace_rhs = lambda assignment: Assignment(
        assignment.lhs, 
        replace_assignments_by_lhs(assignment.rhs)
    )
    
    assignments = map(replace_rhs, assignments)
    if only_unique:
        assignments = iter_unique(assignments, key=hash)
    assignments = sorted(assignments, key=sort_key)
    # assignments = Tuple(*assignments)
    
    #flat_assignments = list(itertools.chain(*map(lambda x:zip_expression(x.lhs, x.rhs, key=is_lvalue), assignments)))

    return assignments

def iter_new_assignments_for_breakouts(expression, symbols = None):
    #assert(type(assignments) == Tuple)
    #assert(all(map(lambda item: type(item) == Assignment), assignments))
    
    recurse = iter_new_assignments_for_breakouts
    result = identity
    
    symbols = unused_symbols(expression, symbols)
    
    if isinstance(expression, BreakOut):
        symbol = next(symbols)
        yield result(Assignment(symbol, expression.expr))
        yield from recurse(expression.expr, symbols)
        
    elif isinstance(expression, Expression):
        for arg in expression.args:
            yield from recurse(arg, symbols)
    
    elif is_mappable_collection(expression):
        for item in expression:
            yield from recurse(item, symbols)


def assign_breakouts(expression, symbols = None):
    symbols = unused_symbols(expression, symbols)

    result = identity
    
    name_hint_symbols = dict()
    
    def next_symbol(name_hint):
        if name_hint == "":
            return next(symbols)
        
        if name_hint not in name_hint_symbols:
            name_hint_symbols[name_hint] = unused_symbols(expression, name_hint)
        
        return next(name_hint_symbols[name_hint])
    
    # reuse already assigned expressions
    assigned = dict()
    def make_assignment(breakout, recurse):
        if breakout not in assigned:
            lhs = next_symbol(breakout.name_hint)
            rhs = recurse(breakout.expr)
            assigned[breakout] = result(Assignment(lhs, rhs))
        return assigned[breakout]
    
    return replace(
        expression,
        BreakOut,
        make_assignment 
    )

def replace_assignments_by_lhs(expression):
    return replace(
        expression, 
        Assignment, 
        lambda expr, recurse: expr.lhs
    )

def replace_assignments_by_rhs(expression):
    return replace(
        expression, 
        Assignment, 
        lambda expr, recurse: expr.rhs
    )
