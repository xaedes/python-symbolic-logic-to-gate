
from functools import partial
from syloga.utils.identity import identity
from syloga.core.is_lvalue import is_lvalue
from syloga.core.map_expression_args import map_expression_args

from syloga.ast.Assignment import Assignment
from syloga.ast.FunctionArgument import FunctionArgument


def replace_lvalues_in_usage_context(expression, replacements, in_usage_context=True, used_replacements=None):
    #echo_display(expression)
    
    recurse = partial(
        replace_lvalues_in_usage_context, 
        replacements=replacements, 
        used_replacements=used_replacements
    )
    result = identity
    
    if is_lvalue(expression) and in_usage_context and expression in replacements:
        if used_replacements is not None: 
            used_replacements.update({expression: replacements[expression]})
        return result(replacements[expression])
    
    elif type(expression) == Assignment:
        return result(Assignment(
            recurse(expression.lhs, in_usage_context=False),
            recurse(expression.rhs, in_usage_context=True)
        ))
    elif type(expression) == FunctionArgument and expression.is_in:
        return result(FunctionArgument(recurse(expression.name, in_usage_context=True)))
    
    elif type(expression) == FunctionArgument and expression.is_out:
        return result(FunctionArgument(recurse(expression.name, in_usage_context=False)))

    else:
        return result(map_expression_args(
            partial(recurse,in_usage_context=in_usage_context),
            expression, 
            recurse_collection=True)
        )

