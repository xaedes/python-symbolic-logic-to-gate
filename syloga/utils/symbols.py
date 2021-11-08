
from syloga.ast.core import Symbol
from syloga.ast.containers import Tuple

def symbols(string):
    return Tuple(*map(Symbol,string.split(" ")))

