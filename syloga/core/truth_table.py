
import itertools
from syloga.transform.to_python import pyfunction_by_repr

def truth_table_by_repr(expression):
    func = pyfunction_by_repr(expression)
    num_args = len(func.args)
    result = [
        (args, func(*args))
        for args in itertools.product([False,True], repeat=num_args)
    ]
    return result
