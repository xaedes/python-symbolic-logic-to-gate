
from syloga.ast.BreakOut import BreakOut
from syloga.transform.replace import replace

def replace_with_breakouts(expression, predicate, *args, **kwargs):
    return replace(expression, predicate, lambda x,r:BreakOut(r(x)), *args, **kwargs)
