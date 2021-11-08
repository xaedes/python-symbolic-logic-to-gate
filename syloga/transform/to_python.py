
from syloga.traversal.iter_symbols import iter_symbols

def pyfunction_by_repr(expression):
    symbols = sorted(iter_symbols(expression),key=str)
    code = "(lambda {args}: ({expression}))".format(
        args = ", ".join(map(str,symbols)),
        expression = repr(expression)
    )
    #print(code)
    function = eval(code)    
    function.args = symbols
    return function
