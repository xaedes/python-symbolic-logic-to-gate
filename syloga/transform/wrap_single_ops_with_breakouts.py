
import itertools

from syloga.ast.Symbol import Symbol
from syloga.ast.BreakOut import BreakOut
from syloga.ast.Assignment import Assignment
from syloga.ast.BooleanExpression import BooleanExpression
from syloga.utils.identity import identity
from syloga.core.map_expression_args import map_expression_args

def wrap_single_ops_with_breakouts(expression, name_hint="o", parent=None): 
    # op_weight_no_symbol = dict(zip(map(id, [Symbol,BreakOut,Assignment]),itertools.repeat(0)))
    skip_breakout = lambda x:(id(type(x)) in set(map(id,[BreakOut,Assignment])))
    include_breakout = lambda x: ( not skip_breakout(x) and isinstance(x,BooleanExpression) )
    #is_single_op = lambda x:(x.count_ops(op_weight_no_symbol) == 1)
    #is_single_op = lambda x:(isinstance(x,Expression) and (x.count_ops(op_weight_no_symbol) > 0))
    
    recurse = lambda x: wrap_single_ops_with_breakouts(x, name_hint, parent=expression)
    result = identity
    if include_breakout(expression) and (not skip_breakout(parent) or parent is None):
        breakout = BreakOut(map_expression_args(recurse, expression), name_hint)
        return result(breakout)
        
    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

