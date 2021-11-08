
import collections.abc

def is_hashable(x):
    """
    Determines whether x is hashable.

    :returns:   True if the specified x is hashable, False otherwise.
    :rtype:     bool
    """
    return hasattr(x, "__hash__") and callable(x.__hash__)


def is_mappable_collection(expression):
    non_collection_types = (str, bytes, bytearray)
    collection_types = (collections.abc.Sequence, dict)
    if isinstance(expression, non_collection_types): return False
    if isinstance(expression, collection_types): return True

def decide_predicate_usage(predicate, default_predicate=False, list_combinator = any):
    p = predicate
    dp = default_predicate
    decide_list = lambda p, dp, x: list_combinator((decide_predicate_usage(p_,dp)(x) for p_ in p))
    is_container = lambda f: isinstance(f, collections.abc.Container)
    return lambda x: (
        p                     if type(p) == bool else
        isinstance(x, p)      if type(p) == type else 
        p(x)                  if callable(p)     else 
        str(x).startswith(p)  if type(p) == str  else
        decide_list(p,dp,x)   if type(p) == list else
        (x in p)              if is_container(p) else
        decide_predicate_usage(dp,dp)(x) if p != dp else
        dp                                                
    )
    