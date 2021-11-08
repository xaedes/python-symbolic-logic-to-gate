import random
import itertools

def compose(*funcs):
    if len(funcs) == 0:
        return identity
    elif len(funcs) == 1:
        return funcs[0]
    else:
        funcs_list = list(funcs)

        def compose_chain(*args, **kwargs): 
            func0 = funcs_list[-1]
            res = func0(*args, **kwargs)
            for func in reversed(funcs_list[:-1]):
                res = func(res)
            
            return res
        
        return compose_chain

def compose_multi(*funcs):
    apply_multi = lambda fs: (lambda *args, **kwargs: (
        ([f(*args, **kwargs) for f in fs] ) if type(fs) == list else ([fs(*args, **kwargs)])
    ))
    if len(funcs) == 0:
        return identity
    elif len(funcs) == 1:
        func0 = funcs[0]
        if callable(func0):
            return func0
        else:
            return apply_multi(func0)
    else:
        funcs_list = list(funcs)

        def apply_funcs(*args, **kwargs): 
            func0 = funcs_list[-1]
            res = apply_multi(func0)(*args, **kwargs)
            for func in reversed(funcs_list[:-1]):
                res = apply_multi(func)(res)
            
            return res
        
        return apply_funcs


def echo(expression,label=None):
    if label is None: print(expression)
    else: print(label, expression)
    return expression


def echo_display(expression,label=None):
    if label is None: display(expression)
    else: display(label, expression)
    return expression


def identity(x, *args, **kwargs):
    return x



def iter_randint(a, b, seed=None, set_seed=True):
    """
    Iterate over a set of random integers in range [a, b], including both end points.

    :param      a:         
    :param      b:         
    :param      seed:      The seed
    :param      set_seed:  Whether to seed the random number generator
    """
    if set_seed: random.seed(seed)
    while True: yield random.randint(a, b)


def iter_unique(iterable, key=str):
    if key is None: key = lambda v:v
    seen = set()
    for item in iterable:
        item_key = key(item)
        if item_key not in seen:
            yield item
            seen.add(item_key)


def list_unique(iterable, key=str):
    if key is None: key = lambda v:v
    items = list(iterable)
    return list(dict(zip(map(key,items), items)).values())


def list_foldl(f, left, xs):
    return (
        left if len(xs) == 0 else
        list_foldl(f, f(left, xs[0]), xs[1:])
    )


def list_foldr(f, xs, right):
    return (
        right if len(xs) == 0 else
        list_foldr(f, f(xs[0], right), xs[1:])
    )


def take_n(iterable, n):
    """
    Yield n items from iterable.

    :param      iterable:  The iterable
    :param      n:         The number of items to yield

    """

    for k in range(n):
        yield next(iterable)

def list_n(iterable, n):
    return list(take_n(iterable, n))


def partition(f, iterable):
    it_True, it_False = itertools.tee(iterable, 2)
    return filter(f, it_True), itertools.filterfalse(f, it_False)

def list_partition(f, iterable):
    return tuple(map(list, partition(f, iterable)))



def map_compose_multi(*funcs):
    apply_multi = lambda fs: (lambda *args, **kwargs: (
        ([f(*args, **kwargs) for f in fs] ) if type(fs) == list else (fs(*args, **kwargs))
    ))
    if len(funcs) == 0:
        return identity
    elif len(funcs) == 1:
        func0 = funcs[0]
        if callable(func0):
            return func0
        else:
            return apply_multi(func0)
    else:
        funcs_list = list(funcs)

        def apply_funcs(*args, **kwargs): 
            func0 = funcs_list[-1]
            res = apply_multi(func0)(*args, **kwargs)
            for func in reversed(funcs_list[:-1]):
                res = list(map(apply_multi(func),res))
            
            return res
        
        return apply_funcs


def star2_compose(*funcs):
    if len(funcs) == 0:
        return identity
    elif len(funcs) == 1:
        return funcs[0]
    else:
        funcs_list = list(funcs)

        def star2_compose_chain(*args, **kwargs): 
            func0 = funcs_list[-1]
            res = func0(*args, **kwargs)
            for func in reversed(funcs_list[:-1]):
                res = func(**res)
            
            return res
        
        return star2_compose_chain

def star_compose(*funcs):
    if len(funcs) == 0:
        return identity
    elif len(funcs) == 1:
        return funcs[0]
    else:
        funcs_list = list(funcs)

        def star_compose_chain(*args, **kwargs): 
            func0 = funcs_list[-1]
            res = func0(*args, **kwargs)
            for func in reversed(funcs_list[:-1]):
                res = func(*res)
            
            return res
        
        return star_compose_chain


def star_compose_multi(*funcs):
    apply_multi = lambda fs: (lambda *args, **kwargs: (
        ([f(*args, **kwargs) for f in fs] ) if type(fs) == list else (fs(*args, **kwargs))
    ))
    if len(funcs) == 0:
        return identity
    elif len(funcs) == 1:
        func0 = funcs[0]
        if callable(func0):
            return func0
        else:
            return apply_multi(func0)
    else:
        funcs_list = list(funcs)

        def apply_funcs(*args, **kwargs): 
            func0 = funcs_list[-1]
            res = apply_multi(func0)(*args, **kwargs)
            for func in reversed(funcs_list[:-1]):
                res = apply_multi(func)(*res)
            
            return res
        
        return apply_funcs
    # compose(list,partial(map,star_compose_multi(
    # lambda k,f,b: str(k) + ":" + ("fizz" if f else "") + ("buzz" if b else ""),
    # [identity,lambda x:x%5==0, lambda x:x%3==0])))(range(30))

