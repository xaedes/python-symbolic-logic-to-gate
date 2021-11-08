
from syloga.ast.core import BooleanAnd
from syloga.ast.core import BooleanNand
from syloga.ast.core import BooleanNor
from syloga.ast.core import BooleanNot
from syloga.ast.core import BooleanOr
from syloga.ast.core import BooleanXor
from syloga.transform.core import map_expression_args
from syloga.utils.predictes import identity

# from syloga.core.assert_equality_by_table import assert_equality_by_table

def replace_by_nand(expression):
    recurse = replace_by_nand
    # result = assert_equality_by_table
    result = identity

    if type(expression) == BooleanOr:
        return result(BooleanNand(*(recurse(BooleanNot(arg)) for arg in expression.args)))
    
    elif type(expression) == BooleanNot:
        assert(len(expression.args) == 1)
        arg = recurse(expression.args[0]);
        return result(BooleanNand(arg, arg))
    
    elif type(expression) == BooleanAnd:
        return result(recurse(BooleanNot(BooleanNand(*expression.args))))
    
    elif type(expression) == BooleanXor:
        assert(len(expression.args) >= 2)
        a, b = expression.args[:2]
        ab = BooleanNand(a, b)
        aab = BooleanNand(a,ab)
        abb = BooleanNand(ab,b)
        aabb = BooleanNand(aab,abb);
        if len(expression.args) > 2:
            aabb = BooleanXor(aabb, *expression.args[2:])
        return recurse(result(aabb))

    elif type(expression) == BooleanNor:
        return result(recurse(BooleanNot(BooleanOr(*expression.args))))

    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

def replace_by_nor(expression):
    recurse = replace_by_nor
    # result = assert_equality_by_table
    result = identity

    if type(expression) == BooleanOr:
        return result(recurse(BooleanNot(BooleanNor(*expression.args))))
        #return result(BooleanNand(*(recurse(BooleanNot(arg)) for arg in expression.args)))
    
    elif type(expression) == BooleanNot:
        assert(len(expression.args) == 1)
        arg = recurse(expression.args[0]);
        return result(BooleanNor(arg, arg))
    
    elif type(expression) == BooleanAnd:
        return result(recurse(BooleanNor(*map(BooleanNot,expression.args))))

    elif type(expression) == BooleanXor:
        assert(len(expression.args) >= 2)
        a, b = expression.args[:2]
        
        na = BooleanNot(a)
        nb = BooleanNot(b)
        nnab = BooleanNor(na, nb)
        nab = BooleanNor(a, b)
        res = BooleanNor(nnab, nab)
        
        if len(expression.args) > 2:
            res = BooleanXor(res, *expression.args[2:])
        return recurse(result(res))
    
    elif type(expression) == BooleanNand:
        return result(recurse(BooleanNot(BooleanAnd(*expression.args))))
    
    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

def replace_by_nor_not_or(expression):
    recurse = replace_by_nor
    # result = assert_equality_by_table
    result = identity

    if type(expression) == BooleanAnd:
        return result(recurse(BooleanNor(*map(BooleanNot,expression.args))))
    
    elif type(expression) == BooleanXor:
        assert(len(expression.args) >= 2)
        a, b = expression.args[:2]
        
        ab = BooleanNor(a, b)
        aab = BooleanNor(a,ab)
        abb = BooleanNor(ab,b)
        aabb = BooleanOr(aab,abb);
        
        if len(expression.args) > 2:
            aabb = BooleanXor(aabb, *expression.args[2:])
        return recurse(result(res))

    elif type(expression) == BooleanNand:
        return result(recurse(BooleanOr(map(BooleanNot,expression.args))))
    
    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))

