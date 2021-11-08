
import itertools
import collections.abc
from functools import partial

from syloga.ast.core import Expression
from syloga.ast.core import Symbol
from syloga.ast.core import Assignment
from syloga.ast.core import is_lvalue
from syloga.ast.core import numbered_symbols
from syloga.ast.containers import Tuple

from syloga.utils.functional import iter_unique
from syloga.utils.functional import list_n

def postorder_traversal(expression):
    #if isinstance(expression, Expression):
    for arg in iter_args(expression):
        yield from postorder_traversal(arg)
    yield expression
    
def preorder_traversal(expression):
    stack = [expression]
    while len(stack)>0:
        item = stack.pop()
        yield item
        for arg in reversed(iter_args(item)):
            stack.append(arg)


def iter_args(expression, default_value = []):
    
    args = (
        list(expression.args)    if isinstance(expression, Expression) else
        default_value            if isinstance(expression, (str, bytes, bytearray)) else
        list(expression)         if isinstance(expression, collections.abc.Sequence) else
        list(expression.items()) if isinstance(expression, dict) else
        default_value
    )
    return args

def iter_expression(expression, default_value = []):
    # TODO: use generator functions
    args = (
        # NOTE: expression.__iter__ may generate new Expressions and may even be infinite (e.g. FunctionCall.__iter__)
        list(expression)         if isinstance(expression, Expression) and hasattr(expression, "__iter__") else 
        list(expression.args)    if isinstance(expression, Expression) else
        default_value            if isinstance(expression, (str, bytes, bytearray)) else
        list(expression)         if isinstance(expression, collections.abc.Sequence) else
        list(expression.items()) if isinstance(expression, dict) else
        default_value
    )
    return args


def iter_lvalues(x, traversal=preorder_traversal):
    return filter(is_lvalue,traversal(x))

def iter_symbols(expression):
    return iter_unique(filter(lambda expr: type(expr) == Symbol, preorder_traversal( expression )))

def list_lvalues(x, traversal=preorder_traversal):
    return list(iter_lvalues(x, traversal=traversal))

def list_symbols(expression, n=None):
    if n is None:
        return list(iter_symbols(expression))
    else:
        return list_n(iter_symbols(expression), n)



def unused_symbols(expression=None, symbols=None):
    if type(symbols) == str: symbols = numbered_symbols(symbols)
    if symbols is None: symbols = numbered_symbols()
    used_symbols = set(iter_symbols(expression)) if expression is not None else set()
    return filter(lambda x:x not in used_symbols, symbols)

def zip_expression(*args, key, flat=False, as_list=True, longest=False, iterator=iter_expression):
    """
    zips recursively until any key(arg) is True.
    
    
    Examples:
    a,b,c = symbols("a b c")
    x,y,z,w = symbols("x y z w")
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:type(x)==Symbol )
    > [(a, x), [(b, y), (c, z)]]
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:type(x)==str )
    > [[('a', 'x')], [[('b', 'y')], [('c', 'z')]]]
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:False )
    > [[[]], [[[]], [[]]]]   (the general structure of the expression zip)
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:True )
    > [(a, x), ((b, c), (y, z))]   (no recursion)

    zip_expression( (a,b), (x,(y,z)), key=lambda x:type(x)==Symbol )
    > [(a, x), (b, (y, z))]
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:type(x)==Symbol, flat=True )
    > [(a, x), (b, y), (c, z)]
    
    zip_expression( (a,b), (x,(y,z)), key=lambda x:type(x)==Symbol, flat=True )
    > [(a, x), (b, (y, z))]
    
    zip_expression( (a,b), (x,y,z), key=lambda x:type(x)==Symbol, longest=True )
    > [(a, x), (b, y), (None, z)]
    
    zip_expression( (a,(b,c)), (x,(y,z,w)), key=lambda x:type(x)==Symbol, longest=True )    
    > [(a, x), [(b, y), (c, z), (None, w)]]
    """
    recurse = partial(zip_expression, key=key, flat=flat, as_list=as_list, longest=longest)
    zip_func = itertools.zip_longest if longest else zip
    if any(map(key,args)):
        gen = iter([args])
    elif flat:
        gen = (
            item
            for tpl in zip_func(*map(iterator,args))
            for item in ([tpl] if any(map(key,tpl)) else recurse(*tpl))
        )
    else:
        gen = (
            tpl if any(map(key,tpl)) else recurse(*tpl)
            for tpl in zip_func(*map(iterator,args))
        )
    
    return list(gen) if as_list else gen
    