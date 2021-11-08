
from syloga.core.map_expression_args import map_expression_args
from syloga.utils.identity import identity

from syloga.ast.BooleanNot import BooleanNot
from syloga.ast.BooleanValue import BooleanValue
from syloga.ast.BooleanOr import BooleanOr
from syloga.ast.BooleanAnd import BooleanAnd
from syloga.ast.BooleanNand import BooleanNand
from syloga.ast.BooleanNor import BooleanNor
from syloga.ast.BooleanXor import BooleanXor
from syloga.ast.BreakOut import BreakOut

# from syloga.core.assert_equality_by_table import assert_equality_by_table

def evaluate_expr(expression):
    recurse = evaluate_expr
    # result = assert_equality_by_table
    result = identity
    #arg_is_value = lambda arg: isinstance(arg, (BooleanValue, bool))
    arg_is_value = lambda arg: type(arg) in [BooleanValue, bool]
    def arg_is_value(arg):
        is_value = type(arg) in [BooleanValue, bool]
        #print("is arg a value? " + str(type(arg)) + " " + str(arg))
        #print("is_value", is_value)
        return is_value
    args_are_values = lambda args: all(map(arg_is_value, args))
    get_value = lambda arg: arg if type(arg) == bool else arg.value
    is_true = lambda val: val == True
    is_false = lambda val: val == False
    
    #print("looking at " + str(type(expression)))
    
    if type(expression) == BooleanNot:
        
        assert(len(expression.args) == 1)
        
        arg = recurse(expression.args[0]);
        if arg_is_value(arg):
            return result(BooleanValue(not get_value(arg)))
        else:
            return result(BooleanNot(arg))
                
    elif type(expression) == BooleanOr:
        args = list(map(recurse, expression.args))
        arg_values = [get_value(arg) for arg in args if arg_is_value(arg)]
        args_wo_neutral = list(filter(lambda x: not(arg_is_value(x) and is_false(get_value(x))),args))
        
        if args_are_values(args):
            return result(BooleanValue(any(arg_values)))
        
        elif any(map(is_true,arg_values)):
            return result(BooleanValue(True))
        
        elif len(args) == 1:
            return result(recurse(args[0]))
        
        elif len(args_wo_neutral) < len(args):
            return result(recurse(BooleanOr(*args_wo_neutral)))
        
        else:
            return result(BooleanOr(*args))
    
    elif type(expression) == BooleanAnd:
        args = list(map(recurse, expression.args))
        #print(expression.args)
        #print(args)
        #negated_atom_values = [not get_value(arg) for arg in args if arg_is_value(arg)]
        arg_values = [get_value(arg) for arg in args if arg_is_value(arg)]
        args_wo_neutral = list(filter(lambda x: not(arg_is_value(x) and is_true(get_value(x))),args))
        #print(arg_values)
        
        if args_are_values(args):
            return result(BooleanValue(all(map(is_true,arg_values))))
        
        elif any(map(is_false,arg_values)):
            return result(BooleanValue(False))
        
        elif len(args) == 1:
            return result(recurse(args[0]))
        
        elif len(args_wo_neutral) < len(args):
            return result(recurse(BooleanAnd(*args_wo_neutral)))

        else:
            return result(BooleanAnd(*args))
    
    elif type(expression) == BooleanNand:
        return result(recurse(BooleanNot(BooleanAnd(*expression.args))))
    
    elif type(expression) == BooleanNor:
        return result(recurse(BooleanNot(BooleanOr(*expression.args))))
    
    elif type(expression) == BooleanXor:
        args = list(map(recurse, expression.args))
        
        arg_values = [get_value(arg) for arg in args if arg_is_value(arg)]
        non_value_args = [arg for arg in args if not arg_is_value(arg)]
        
        if len(args) == 0:
            raise ValueError("args are missing")
            
        elif len(args) == 1:
            return result(args[0])
        
        elif len(arg_values) == 0:
            return result(BooleanXor(*non_value_args))
        
        elif len(arg_values) == 1:
            if is_true(arg_values[0]):
                return result(BooleanXor(arg_values[0], *non_value_args))
            else:
                return result(recurse(BooleanXor(*non_value_args)))
            
        elif len(arg_values) > 1:
            evaluated = is_true(arg_values[0])
            for a in arg_values[1:]:
                evaluated ^= is_true(a)
            evaluated = bool(evaluated)
            
            return result(recurse(BooleanXor(evaluated, *non_value_args)))
    
    
    elif type(expression) == BreakOut:
        expr = recurse(expression.expr)
        if arg_is_value(expr):
            return result(BooleanValue(expr))
        else:
            return result(BreakOut(expr))
    
    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

