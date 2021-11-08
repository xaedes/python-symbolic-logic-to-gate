
from syloga.ast.containers import Tuple
from syloga.ast.core import Expression
from syloga.ast.core import BooleanValue
from syloga.ast.traversal import iter_args
from syloga.utils.functional import identity
from syloga.utils.predicates import is_mappable_collection

def to_expr(argument):
    recurse = expr
    result = lambda x:x
    
    if type(argument) == bool:
        return result(BooleanValue(argument))
    
    elif type(argument) == tuple:
        return result(Tuple(*argument))
    
    return result(map_collection_or_pass(recurse, argument))

def map_args(func, expression):
    return map(func, iter_args(expression))

def map_collection(func, expression):
    return type(expression)(map_args(func, expression))

def map_collection_or_pass(func, expression):
    if is_mappable_collection(expression):
        return map_collection(func, expression)
    else:
        return expression

def map_expression_args(function, expression, recurse_collection=True):
    if isinstance(expression, Expression):
        #try:
        return expression.func(*map(function, expression.args))
        #except:
            #print(expression)
            #raise
    
    elif recurse_collection:
        return map_collection_or_pass(function, expression)
    
    else:
        return expression

def replace(expression, needle, replacement, pre_recurse=identity, post_recurse=identity, result=identity, *args, **kwargs):
    def recurse(expr): 
        return post_recurse(
            replace(
                pre_recurse(expr), 
                needle, 
                *args, 
                replacement=replacement,
                pre_recurse=pre_recurse, 
                post_recurse=post_recurse, 
                result=result, 
                **kwargs
            )
        )
        
    if type(replacement) == type:
        replacement = lambda expr, recurse: replacement(*map(recurse,expr.args))
        
    if (type(needle) == type) and isinstance(expression, needle):
        return result(replacement(expression, recurse))
    
    elif (type(needle) != type) and callable(needle) and needle(expression):
        return result(replacement(expression, recurse))
    
    elif isinstance(expression, list):
        return result(Tuple(*map(recurse,expression)))

    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

def replace_with_context(expression, needle, replacement, context=None, new_context=None, pre_recurse=identity, post_recurse=identity, result=identity, *args, **kwargs):
    if new_context is None: new_context = lambda x,c: x
    def recurse(expr): 
        return post_recurse(
            replace_with_context(
                pre_recurse(expr), 
                needle, 
                *args, 
                context=new_context(expression, context),
                replacement=replacement,
                pre_recurse=pre_recurse, 
                post_recurse=post_recurse, 
                result=result, 
                **kwargs
            )
        )
    
    if type(replacement) == type:
        replacement = lambda expr, recurse, context: replacement(*map(recurse,expr.args))
        
    if (type(needle) == type) and isinstance(expression, needle):
        return result(replacement(expression, recurse, context))
    
    elif (type(needle) != type) and callable(needle) and needle(expression, context):
        return result(replacement(expression, recurse, context))
    
    elif isinstance(expression, list):
        return result(Tuple(*map(recurse,expression)))

    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

