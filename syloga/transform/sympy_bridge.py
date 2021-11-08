
from syloga.ast.core import Assignment
from syloga.ast.core import BooleanValue
from syloga.ast.core import Expression
from syloga.ast.core import Symbol
from syloga.ast.containers import Tuple
from syloga.transform.basic import map_collection_or_pass
from syloga.ast.string_dict import string_dict
from syloga.utils.identity import identity

import sympy
import sympy.codegen.ast

def expr_to_sympy(expression):
    recurse = expr_to_sympy
    result = identity
    
    if type(expression) == Symbol:
        return result(sympy.Symbol(expression.name))
    
    if type(expression) == Tuple:
        return result(sympy.Tuple(*map(recurse, expression.args)))
    
    if type(expression) == Assignment:
        return result(sympy.codegen.ast.Assignment(*map(recurse, expression.args)))
    
    elif type(expression) == BooleanValue:
        return result(sympy.true if expression.value == True else sympy.false)
    
    # todo: 
    # FunctionArgument
    # FunctionDefinition (sympy.codegen.ast.FunctionPrototype)
    # FunctionDeclaration (sympy.codegen.ast.FunctionDefinition)
    # FunctionCall
    
    
    elif isinstance(expression, Expression):
        return result(sympy.Function(expression.func.__name__)(*map(recurse, expression.args)))
    
    else:
        return result(map_collection_or_pass(recurse, expression))

def sympy_to_expr(expression):
    recurse = sympy_to_expr
    result = identity

    if type(expression) == sympy.Symbol:
        return result(Symbol(expression.name))
    
    if type(expression) == sympy.Tuple:
        return result(Tuple(*map(recurse,expression.args)))
    
    if type(expression) == sympy.codegen.ast.Assignment:
        return result(Assignment(*map(recurse,expression.args)))
    
    elif type(expression) == sympy.logic.boolalg.BooleanTrue:
        return result(BooleanValue(True))
        
    elif type(expression) == sympy.logic.boolalg.BooleanFalse:
        return result(BooleanValue(False))
        
    elif isinstance(expression, sympy.core.function.AppliedUndef):
        global string_dict
        if expression.name in string_dict:
            return result(string_dict[expression.name](*map(recurse, expression.args)))
        else:
            return result(Function(expression.name, *map(recurse, expression.args)))
        return result(sympy.Function(expression.func.__name__)(*map(recurse, expression.args)))
    
    # todo: 
    # FunctionArgument
    # FunctionDefinition (sympy.codegen.ast.FunctionPrototype)
    # FunctionDeclaration (sympy.codegen.ast.FunctionDefinition)
    # FunctionCall

    
    elif isinstance(expression, Expression):
        return result(expression.func(*map(recurse,expression.args)))
    
    elif isinstance(expression, list):
        return result(Tuple(*map(recurse,expression)))

    else:
        return result(map_collection_or_pass(recurse, expression))
