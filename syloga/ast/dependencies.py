
from syloga.ast.component import ComponentSection
from syloga.ast.component import ExpressionComponent
from syloga.ast.core import Assignment
from syloga.ast.core import Expression
from syloga.ast.core import is_lvalue
from syloga.ast.function import FunctionCall
from syloga.utils.predicates import is_mappable_collection

def iter_dependees(x,in_lhs=False):
    """
    free lvalues on lhs, bound output arguments of function calls (on lhs and rhs) and outputs of expression components

    may include duplicates, use iter_unique to filter those out.
    
    list(iter_dependees(Symbol("a")))
    > []

    list(iter_dependees(Assignment(Symbol("a"),Symbol("b"))))
    > [a]

    list(iter_dependees(Assignment(Tuple(Symbol("a"),Symbol("b")),Symbol("c"))))
    > [a, b]

    list(iter_dependees(FunctionDeclaration("f",(FunctionArgument.In(Symbol("a")),FunctionArgument.Out(Symbol("b")))).call(*symbols("x y"))))
    > [y]
    
    list(iter_dependees(FunctionDeclaration("f",(FunctionArgument.In(Symbol("a")),FunctionArgument.Out(Symbol("b")))).call(*symbols("x"))))
    > []
    """

    recurse = iter_dependees

    if type(x) == Assignment:
        yield from recurse(x.lhs, in_lhs=True)
        yield from recurse(x.rhs, in_lhs=False)
    elif in_lhs and is_lvalue(x):
        yield x
    elif type(x) == FunctionCall:
        yield from x.bound_output_arguments().values()
    elif type(x) == ExpressionComponent:
        yield from filter(is_lvalue, x.section_symbols[ComponentSection.Output])
    elif type(x) == Tuple or (is_mappable_collection(x) and not isinstance(x,dict)):
        for y in x:
            yield from recurse(y, in_lhs=in_lhs)
    else:
        return

def iter_dependencies(x,parent=None,in_rhs=True):
    """
    free lvalues on rhs, bound input arguments of function calls (on lhs and rhs) and inputs of expression components.
    
    may include duplicates, use iter_unique to filter those out.

    list(iter_dependencies(Symbol("a")))
    > [a]

    list(iter_dependencies(Assignment(Symbol("a"),Symbol("b"))))
    > [b]

    list(iter_dependencies(Assignment(Tuple(Symbol("a"),Symbol("b")),Symbol("c"))))
    > [c]

    list(iter_dependencies(FunctionDeclaration("f",(FunctionArgument.In(Symbol("a")),FunctionArgument.Out(Symbol("b")))).call(*symbols("x y"))))
    > [x]
    
    list(iter_dependencies(FunctionDeclaration("f",(FunctionArgument.In(Symbol("a")),FunctionArgument.Out(Symbol("b")))).call(*symbols("x"))))
    > [x]
    
    """
    #print(x, parent)
    recurse = iter_dependencies

    if type(x) == Assignment:
        yield from recurse(x.rhs, parent=x, in_rhs=True)
        yield from recurse(x.lhs, parent=x, in_rhs=False)
    elif in_rhs and is_lvalue(x) and not (type(parent) == FunctionArgument and parent.is_out):
        yield x
    elif type(x) == FunctionCall:
        yield from x.bound_input_arguments().values()
    elif type(x) == ExpressionComponent:
        yield from filter(is_lvalue, x.section_symbols[ComponentSection.Input])
    elif type(x) == Tuple or (is_mappable_collection(x) and not isinstance(x,dict)):
        for y in x:
            yield from recurse(y, parent=x, in_rhs=in_rhs)
    elif isinstance(x, Expression):
        for y in x.args:
            yield from recurse(y, parent=x, in_rhs=in_rhs)
    else:
        return
