#!/usr/bin/env python
# coding: utf-8

# In[269]:


import sys
print("Python Version")
print(sys.version)
#%matplotlib ipympl
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib widget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import os
from collections import defaultdict
from collections import OrderedDict
import numba
import sympy as sp
import sympy.codegen as cgen
import sympy.abc as abc
import functools
from sympy.core.basic import preorder_traversal
from sympy.utilities.iterables import postorder_traversal
import sympy.codegen.rewriting
import sympy.codegen.ast as ast
from enum import Enum
import dataclasses
from dataclasses import dataclass
import itertools
from IPython.display import clear_output

import typing
import moderngl
import random
import collections
import re
import operator
from functools import partial

def is_hashable(x):
    return hasattr(x, "__hash__") and callable(x.__hash__)

def take_n(iterable, n):
    for k in range(n):
        yield next(iterable)

def list_n(iterable, n):
    return list(take_n(iterable, n))

def list_unique(iterable, key=str):
    if key is None: key = lambda v:v
    items = list(iterable)
    return list(dict(zip(map(key,items), items)).values())

def iter_unique(iterable, key=str):
    if key is None: key = lambda v:v
    seen = set()
    for item in iterable:
        item_key = key(item)
        if item_key not in seen:
            yield item
            seen.add(item_key)

def partition(f, iterable):
    it_True, it_False = itertools.tee(iterable, 2)
    return filter(f, it_True), itertools.filterfalse(f, it_False)

def list_partition(f, iterable):
    return tuple(map(list, partition(f, iterable)))

def iter_randint(a,b, seed=None, set_seed=True):
    if set_seed: random.seed(seed)
    while True: yield random.randint(a,b)

def iter_args(expression, default_value = []):
        
    args = (
        list(expression.args)    if isinstance(expression, Expression) else
        default_value            if isinstance(expression, (str, bytes, bytearray)) else
        list(expression)         if isinstance(expression, collections.abc.Sequence) else
        list(expression.items()) if isinstance(expression, dict) else
        default_value
    )
    return args

def iter_expression(expression, default_value = []):
        
    args = (
        # NOTE: expression.__iter__ may generate new Expressions and may even be infinite (e.g. FunctionCall.__iter__)
        list(expression)         if isinstance(expression, Expression) and hasattr(expression, "__iter__") else 
        list(expression.args)    if isinstance(expression, Expression) else
        default_value            if isinstance(expression, (str, bytes, bytearray)) else
        list(expression)         if isinstance(expression, collections.abc.Sequence) else
        list(expression.items()) if isinstance(expression, dict) else
        default_value
    )
    return args

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
            print(funcs_list[:-1])
            
            for func in reversed(funcs_list[:-1]):
                res = func(*res)
            
            return res
        
        return star_compose_chain

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

def try_or_default(default_value, func):
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return default_value
    return wrap

def try_or_false(func):
    return try_or_default(False, func)
def try_or_true(func):
    return try_or_default(True, func)

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

def preorder_traversal(expression):
    stack = [expression]
    while len(stack)>0:
        item = stack.pop()
        yield item
        for arg in reversed(iter_args(item)):
            stack.append(arg)

def postorder_traversal(expression):
    #if isinstance(expression, Expression):
    for arg in iter_args(expression):
        yield from postorder_traversal(arg)
    yield expression

@dataclass
class Expression:
    func: type
    args: tuple
    def __init__(self, func, args):
        self.func = func
        self.args = tuple(args)
    
    def __str__(self):
        #return "{type}[{repr}]".format(
        #    type = str(type(self).__name__),
        #    repr = repr(self)
        #)
        #str(type(self)) + ":" + repr(self)
        return repr(self) 
    
    def __hash__(self):
        return hash(tuple((id(self.func), tuple(list(map(hash,self.args))))))
        #return hash(str(self))
    
    def __iter__(self):
        return iter(self.args)
    
    def __and__(self, other):
        return BooleanAnd(self, other)
    
    def __or__(self, other):
        return BooleanOr(self, other)
    
    def __xor__(self, other):
        return BooleanXor(self, other)
    
    def __invert__(self):
        return BooleanNot(self)
    
    def count_ops(self, op_weight = None):
        recurse = lambda x: x.count_ops(op_weight) if isinstance(x,Expression) else 0
        count = sum(map(recurse, self.args))
        if op_weight is None: 
            count += 1
        elif id(type(self)) in op_weight:
            count += op_weight[id(type(self))]
        else:
            count += 1
        return count

@dataclass
class Symbol(Expression):
    name: str
    def __init__(self, name:str):
        self.name = str(name)
        super().__init__(Symbol, (self.name, ))
        
    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return self.name

@dataclass
class Indexed(Expression):
    name: Expression
    index: int

    def __init__(self, name:Expression, index: int):
        self.name = name
        self.index = int(index)
        super().__init__(Indexed, (self.name, self.index))
        
    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return "%s[%s]" % (str(self.name), str(self.index))

    def __getitem__(self, index:int):
        return Indexed(self, index)

@dataclass
class Indexable(Symbol):
    def __init__(self, name:str):
        super().__init__(name)
        
    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return self.name
    
    def __getitem__(self, index:int):
        return Indexed(self, index)

@dataclass
class Tuple(Expression):
    def __init__(self, *args):
        super().__init__(Tuple, args)
    
    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return repr(self.args)
    
    def __getitem__(self, k):
        return self.args[k]
    
    def __iter__(self):
        return iter(self.args)
    
    def __len__(self):
        return len(self.args)

@dataclass
class Dict(Expression):
    def __init__(self, *args, **kwargs):
        self.dictionary = dict(args)
        self.dictionary.update(kwargs)
        super().__init__(Dict, tuple(self.dictionary.items()))
    
    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        #return repr(self.dictionary)
        return "Dict%s" % repr(self.dictionary)
    
    def __getitem__(self, k):
        return self.dictionary[k]
    
    def __contains__(self, k):
        return k in self.dictionary
    
    def __iter__(self):
        return iter(self.dictionary)
    
    def keys(self):
        return Tuple(*self.dictionary.keys())
    
    def values(self):
        return Tuple(*self.dictionary.values())
    
    def items(self):
        return Tuple(*self.args)
    
    def update(self, *E, **F):
        self.dictionary.update(*E,**F)
        self.args = self.items()
    
    def __len__(self):
        return len(self.dictionary)


# In[38]:


Dict(**{'a':1,'b':2})


# In[39]:


help(dict.update)


# In[40]:


@dataclass
class Assignment(Expression):
    lhs: Expression
    rhs: Expression
    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs
        super().__init__(Assignment, (self.lhs, self.rhs))
    
    def __hash__(self):
        return super().__hash__()
    
    def __iter__(self):
        return iter(self.args)
    
    def __repr__(self):
        return "[%s = %s]" % (self.lhs, self.rhs)

@dataclass
class BreakOut(Expression):
    """Assign the given expression later with an explicit symbol"""
    expr: Expression
    name_hint: str
    expose: bool

    def __init__(self, expr: Expression, name_hint: str = "", expose:bool = True):
        self.expr = expr
        self.name_hint = name_hint
        self.expose = bool(expose)
        super().__init__(BreakOut, (self.expr, self.name_hint, self.expose))
    
    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return "%s#%s#=%s" % ("<" if self.expose else "",self.name_hint, self.expr)

@dataclass
class Function(Expression):
    name: str
    def __init__(self, name:str, *args):
        self.name = name
        super().__init__(Function, (name, args))

    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return self.name
    
@dataclass
class BooleanExpression(Expression):
    pass
    #def __init__(self, *args, **kwargs):
    #    super().__init__(self, *args, **kwargs)
    def __hash__(self):
        return super().__hash__()
    
    def __and__(self, other):
        return BooleanAnd(self, other)
    
    def __or__(self, other):
        return BooleanOr(self, other)
    
    def __xor__(self, other):
        return BooleanXor(self, other)
    
    def __invert__(self):
        return BooleanNot(self)

@dataclass
class BooleanAnd(BooleanExpression):
    def __init__(self, *args):
        super().__init__(BooleanAnd, args)

    def __hash__(self):
        return super().__hash__()
        
    def __repr__(self):
        return "(" + " and ".join(map(repr, self.args)) + ")"

@dataclass
class BooleanNand(BooleanExpression):
    def __init__(self, *args):
        super().__init__(BooleanNand, args)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return "not (" + " and ".join(map(repr, self.args)) + ")"

@dataclass
class BooleanNor(BooleanExpression):
    def __init__(self, *args):
        super().__init__(BooleanNor, args)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return "not (" + " or ".join(map(repr, self.args)) + ")"

@dataclass
class BooleanOr(BooleanExpression):
    def __init__(self, *args):
        super().__init__(BooleanOr, args)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return "(" + " or ".join(map(repr, self.args)) + ")"


@dataclass
class BooleanXor(BooleanExpression):
    def __init__(self, *args):
        super().__init__(BooleanXor, args)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return "(" + " ^ ".join(map(repr, self.args)) + ")"


@dataclass
class BooleanNot(BooleanExpression):
    arg: Expression
    def __init__(self, arg: Expression):
        self.arg = arg
        super().__init__(BooleanNot, (arg,))

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return "not " + repr(self.arg)

@dataclass
class BooleanValue(BooleanExpression):
    value: bool
    def __init__(self, value: bool):
        self.value = value
        super().__init__(BooleanValue, (self.value, ))

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return repr(self.value)
    
    def __bool__(self):
        return bool(self.value)
    
@dataclass
class FunctionArgument(Expression):
    name: Expression
    is_in: bool
    is_out: bool
    def __init__(self, name, is_in=True, is_out=False):
        self.name = name
        self.is_in = is_in
        self.is_out = is_out
        assert((is_in or is_out) == True)
        assert((is_in and not is_out) or (not is_in and is_out))
        # so far i don't know how to use inout (only in xor out), but it seems useful to keep for future use
        super().__init__(FunctionArgument, (self.name, self.is_in, self.is_out))
        
    def In(name):
        return FunctionArgument(name, is_in=True, is_out=False)

    def Out(name):
        return FunctionArgument(name, is_in=False, is_out=True)
        
    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return (
            ("in"      if self.is_in  else "")
          + ("out"     if self.is_out else "")
          + (" "       if self.is_in or self.is_out else "")
          + (self.name if type(self.name) == str    else repr(self.name))
        )

def zip_expression(*args, key, flat=False, as_list=True, longest=False, iterator=iter_expression):
    """
    zips recursively until any key(arg) is True.
    
    
    Examples:
    a,b,c = symbols("a b c")
    x,y,z,w = symbols("x y z w")
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:type(x)==Symbol )
    > [(a, x), [(b, y), (c, z)]]
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:type(x)==str )
    > [[('a', 'x')], [[('b', 'y')], [('c', 'z')]]]
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:False )
    > [[[]], [[[]], [[]]]]   (the general structure of the expression zip)
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:True )
    > [(a, x), ((b, c), (y, z))]   (no recursion)

    zip_expression( (a,b), (x,(y,z)), key=lambda x:type(x)==Symbol )
    > [(a, x), (b, (y, z))]
    
    zip_expression( (a,(b,c)), (x,(y,z)), key=lambda x:type(x)==Symbol, flat=True )
    > [(a, x), (b, y), (c, z)]
    
    zip_expression( (a,b), (x,(y,z)), key=lambda x:type(x)==Symbol, flat=True )
    > [(a, x), (b, (y, z))]
    
    zip_expression( (a,b), (x,y,z), key=lambda x:type(x)==Symbol, longest=True )
    > [(a, x), (b, y), (None, z)]
    
    zip_expression( (a,(b,c)), (x,(y,z,w)), key=lambda x:type(x)==Symbol, longest=True )    
    > [(a, x), [(b, y), (c, z), (None, w)]]
    """
    recurse = partial(zip_expression, key=key, flat=flat, as_list=as_list, longest=longest)
    zip_func = itertools.zip_longest if longest else zip
    if any(map(key,args)):
        gen = iter([args])
    elif flat:
        gen = (
            item
            for tpl in zip_func(*map(iterator,args))
            for item in ([tpl] if any(map(key,tpl)) else recurse(*tpl))
        )
    else:
        gen = (
            tpl if any(map(key,tpl)) else recurse(*tpl)
            for tpl in zip_func(*map(iterator,args))
        )
    
    return list(gen) if as_list else gen
    
    #iter_args(a)


# In[ ]:





# In[47]:


def foo(**kwargs):
    print(kwargs)


# In[48]:


Dict(*Dict((1,2),(3,4)).items())


# In[49]:


@dataclass
class FunctionOutput(Expression):
    function_call: Expression
    function_argument: Expression
    def __init__(self, function_call: Expression, function_argument: Expression):
        # designed for this purposes, redesign if asserts are problematic
        assert(type(function_call) == FunctionCall)
        assert(type(function_argument) == FunctionArgument)
        self.function_call = function_call
        self.function_argument = function_argument
        super().__init__(FunctionOutput, (self.function_call, self.function_argument))

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return (
            repr(self.function_call)
          + "["
          + repr(self.function_argument)
          + "]"
        )


# In[50]:


@dataclass
class FunctionCall(Expression):
    function_declaration: Expression
    arguments: Tuple
    kwarguments: Dict
    def __init__(self, function_declaration, arguments=Tuple(), kwarguments={}, assert_valid=True):
        self.function_declaration = function_declaration
        self.arguments = Tuple(*arguments)
        self.kwarguments = Dict(*kwarguments.items())
        if assert_valid:
            try:
                assert(self.arguments_valid())
            except AssertionError:
                print(self.arguments)
                raise
            try:
                assert(self.kwarguments_valid())
            except AssertionError:
                print(self.kwarguments)
                raise
        
        super().__init__(FunctionCall, (self.function_declaration, self.arguments, self.kwarguments))
        
    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return (
            ""
          + str(self.function_declaration.name)
          + "(" 
          + ", ".join(itertools.chain(
                map(str, self.arguments),
                map(lambda item: "=".join(map(str,item)),self.kwarguments.items()),
                (["..."] if self.is_partial() else [])
            ))
          + ")"
        )
    
    def __iter__(self):
        # this allows structural binding by assignment
        return iter(map(lambda x:FunctionOutput(self,x),self.unbound_output_arguments()))
    
    def arguments_valid(self):
        return len(self.arguments) <= len(self.function_declaration.arguments)
    
    def kwarguments_valid(self):
        if len(self.kwarguments) > len(self.function_declaration.arguments) - len(self.arguments): return False
        declaration_args = set((arg.name for arg in self.function_declaration.arguments[len(self.arguments):]))
        for kw,arg in self.kwarguments.items():
            if kw not in declaration_args: return False
        return True
    
    def is_valid(self):
        return self.arguments_valid() and self.kwarguments_valid()
    
    def is_partial(self):
        if not self.is_valid(): return False
        return not self.is_complete()
    
    def is_complete(self):
        if not self.is_valid(): return False
        declaration_args = set((arg.name for arg in self.function_declaration.arguments[len(self.arguments):]))
        return all((
            arg.name in self.kwarguments
            for arg in self.function_declaration.arguments[len(self.arguments):]
        ))
    
    def unbound_arguments(self, with_idx = False):
        num_bound_pos_args = len(self.arguments)
        declaration_args = set((arg.name for arg in self.function_declaration.arguments[num_bound_pos_args:]))
        return [
            ((num_bound_pos_args+k, arg) if with_idx else arg)
            for k,arg in enumerate(self.function_declaration.arguments[num_bound_pos_args:])
            if arg.name not in self.kwarguments
        ]
    
    def unbound_output_arguments(self, with_idx = False):
        return Tuple(
            *filter(
                lambda x:(x[1].is_out if with_idx else x.is_out),
                self.unbound_arguments(with_idx=with_idx)
            )
        )
    
    def bound_arguments(self):
        
        num_bound_pos_args = len(self.arguments)
        return Dict(*itertools.chain(
            zip(self.function_declaration.arguments, self.arguments),
            #enumerate(self.function_declaration.arguments[:num_bound_pos_args]) if with_idx else self.function_declaration.arguments[:num_bound_pos_args],
            #enumerate(self.arguments) if with_idx else self.arguments,
            (
                #((num_bound_pos_args+k, arg) if with_idx else arg)
                (arg, self.kwarguments[arg])
                for arg in self.function_declaration.arguments[num_bound_pos_args:]
                if arg.name in self.kwarguments
            )
        ))

    def bound_input_arguments(self):
        return Dict(
            *filter(
                lambda x:x[0].is_in,
                self.bound_arguments().items()
            )
        )
        
    def bound_output_arguments(self, with_idx = False):
        return Dict(
            *filter(
                lambda x:x[0].is_out,
                self.bound_arguments().items()
            )
        )
    

    
    def get_dependees(self):
        return list(filter(lambda x:x.is_out,self.arguments))
    
    def get_dependencies(self):
        return list(filter(lambda x:x.is_in,self.arguments))    
    
    def inline(self, function_definition):
        assert(function_definition.function_declaration == self.function_declaration)
        return function_definition.inline(self)
    
    def assign_to(self, lhs):
        """make new FunctionCall with lhs assigned to unbound output arguments."""
        kwarguments = dict(*self.kwarguments.items())
        kwarguments.update(
            zip_expression(
                map_expression_args(
                    lambda x:x.name,
                    self.unbound_output_arguments()
                ),
                lhs,
                key=decide_predicate_usage([Symbol, Indexed, Indexable]),
                flat=True,
                as_list=False
            )        
        )
        return FunctionCall(
            self.function_declaration,
            Tuple(*self.arguments),
            kwarguments
        )
        


# In[51]:


@dataclass
class FunctionDeclaration(Expression):
    name: str
    arguments: Tuple
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = Tuple(*arguments)
        super().__init__(FunctionDeclaration, (self.name, self.arguments))
        
    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return (
            ""
          + str(self.name)
          + "[" 
          + repr(self.arguments)[1:-1]
          + "]"
        )
    
    def get_dependees(self):
        return list(filter(lambda x:x.is_out,self.arguments))
    
    def get_dependencies(self):
        return list(filter(lambda x:x.is_in,self.arguments))
    
    def call(self, *args, **kwargs):
        return FunctionCall(self, args, kwargs)

FunctionDeclaration("foo", (FunctionArgument.Out(Symbol("a")),))FunctionDeclaration("foo", (FunctionArgument.Out(Symbol("a")),)).call(1).bound_output_arguments().values()FunctionDeclaration("foo", (FunctionArgument.In(Symbol("a")), FunctionArgument.In(Symbol("b")), FunctionArgument.Out(Symbol("res")),)).call(*symbols("x y")).assign_to((Symbol("z"),))a,b,c,x,y,z,w = symbols("a b c x y z w")zip_expression((a,), (b,c), key=lambda x:type(x)==Symbol)w = Symbol("w")zip_expression( (a,(b,c)), (x,(y,z,w)), key=lambda x:type(x)==Symbol, longest=True )zip_expression(
    *Assignment(
        Tuple(Symbol("x")),
        FunctionDeclaration("foo", (FunctionArgument.Out(Symbol("a")),)).call().unbound_output_arguments()
    ).args,
    key=decide_predicate_usage(Symbol,False)
)

# In[52]:


@dataclass
class FunctionDefinition(Expression):
    function_declaration : Expression
    body: Expression
    def __init__(self, function_declaration:FunctionDeclaration, body:Expression):
        self.function_declaration = function_declaration
        self.body = body
        super().__init__(FunctionDefinition, (self.function_declaration, self.body))
        
    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return (
            ""
          + repr(self.function_declaration)
          + " := "
          + repr(self.body)
        )
    
    def call(self, *args, **kwargs):
        return self.function_declaration.call(*args, **kwargs)
    
    def inline(self, call):
        if type(call) == Assignment: return self.inline_assignment(call)
        if (type(call) != FunctionCall) and isinstance(call, Expression):
            return replace(
                call,
                needle=Assignment,
                replacement=lambda x,r:(
                    self.inline(x) 
                    if type(x.rhs) == FunctionCall and x.rhs.function_declaration == self.function_declaration
                    else map_expression_args(r,x)
                )
            )
        assert(call.function_declaration == self.function_declaration)
        all_symbols = [arg.name for arg in self.function_declaration.arguments]
        symbol_idx = dict(itertools.starmap(lambda k,x: (x,k), enumerate(all_symbols)))
        
        
        arg_replacements = (zip(all_symbols, call.arguments))
        kwarg_replacements = (
            (all_symbols[symbol_idx[kw]], arg)
            for kw,arg in call.kwarguments.items()
            if kw in symbol_idx
        )
        replacements = dict(itertools.chain(arg_replacements, kwarg_replacements))
        
        return replace(
            self.body,
            lambda x: is_hashable(x) and (x in replacements),
            lambda x,r: replacements[x]
        )
    
    def inline_assignment(self, assignment):
        call = assignment.rhs
        assert(type(call) == FunctionCall)
        return self.inline(call.assign_to(assignment.lhs))


# In[53]:


FunctionDefinition(
    FunctionDeclaration("foo", (FunctionArgument.In(Symbol("a")), FunctionArgument.In(Symbol("b")), FunctionArgument.Out(Symbol("out"))) ),
    Assignment( Symbol("out"), BooleanAnd(Symbol("a"), Symbol("b")) )
)


# In[54]:


FunctionDefinition(
    FunctionDeclaration("foo", (FunctionArgument.In(Symbol("a")), FunctionArgument.In(Symbol("b")), FunctionArgument.Out(Symbol("out"))) ),
    Assignment( Symbol("out"), BooleanAnd(Symbol("a"), Symbol("b")) )
).call(True)

FunctionDeclaration(
    "foo", (FunctionArgument.In(Symbol("a")), FunctionArgument.In(Symbol("b")), FunctionArgument.Out(Symbol("out"))) 
).call(True).inline(
    FunctionDefinition(
        FunctionDeclaration("foo", (FunctionArgument.In(Symbol("a")), FunctionArgument.In(Symbol("b")), FunctionArgument.Out(Symbol("out"))) ),
        Assignment( Symbol("out"), BooleanAnd(Symbol("a"), Symbol("b")) )
    )
)SymbolSections
ExpressionSections
> ExpressionComponent
ComponentSections
ComponentSymbols
ComponentSectionSymbolsMemorySections
> MemoryComponent
MemoryAssignmentComponent
ComponentMemory
ComponentMemoryLayout
LayoutedComponent
LinkedComponent
Component
# In[55]:


class ComponentSection(Enum):
    Input = 0
    Output = 1
    Local = 2
    Data = 3
    Rest = 4


# In[56]:


# SymbolSections
# ExpressionSections
# > ExpressionComponent
# ComponentSections
# ComponentSymbols
# ComponentSectionSymbols
# 
# MemorySections
# > MemoryComponent
# MemoryAssignmentComponent
# ComponentMemory
# ComponentMemoryLayout
# LayoutedComponent
# LinkedComponent
# Component

@dataclass
class ExpressionComponent(Expression):
    section_symbols:Dict # Dict is ordered
    expr:Expression
    def __init__(self, section_symbols:Dict, expr:Expression=None):
        self.section_symbols = Dict(*section_symbols.items())
        self.expr = expr
        super().__init__(ExpressionComponent, (self.section_symbols, self.expr))

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return ("ExpressionComponent" + repr(self.args))
    
    def assign(self, *args, **kwargs):
        all_symbols = list(itertools.chain(*self.section_symbols.values()))
        symbol_idx = dict(itertools.starmap(lambda k,x: (str(x),k), enumerate(all_symbols)))
        
        arg_replacements = (zip(all_symbols, args))
        kwarg_replacements = (
            (all_symbols[symbol_idx[kw]], arg)
            for kw,arg in kwargs.items()
            if kw in symbol_idx
        )
        replacements = dict(itertools.chain(arg_replacements, kwarg_replacements))
        
        return replace(
            self,
            Symbol,
            lambda x,r: x if x not in replacements else replacements[x]
        )
    
    def function_declaration(self, name, sections=[ComponentSection.Input, ComponentSection.Output]):
        section_symbol_to_argument = lambda section, symbol: FunctionArgument(
            symbol, 
            is_in = section != ComponentSection.Output,
            is_out = section == ComponentSection.Output
        )
        section_symbols = lambda section: self.section_symbols[section] if section in self.section_symbols else []
        section_arguments = lambda section: map(partial(section_symbol_to_argument, section), section_symbols(section))
        all_arguments = itertools.chain(*map(section_arguments, sections))
        return FunctionDeclaration(
            name = name, 
            arguments = Tuple(*all_arguments)
        )
    
    def function_definition(self, name, sections=[ComponentSection.Input, ComponentSection.Output]):
        return FunctionDefinition(self.function_declaration(name,sections), self)
    
    def call(self, *args, **kwargs):
        # convert all args to kwargs
        all_symbols = list(itertools.chain(*self.section_symbols.values()))
        argdict = dict(zip(map(str,all_symbols), args))
        argdict.update(kwargs)
        return self.function_declaration.call(**argdict)
        
    
@dataclass
class MemoryComponent(Expression):
    component:ExpressionComponent
    address_map:dict
    
    def __init__(self, component:ExpressionComponent, address_map:dict):
        self.component = component
        self.address_map = address_map
        super().__init__(MemoryComponent, (self.component, self.address_map))

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return repr("MemoryComponent" + repr(self.args))

#type
bkcla2 = compose(
    #len,
    #list,
    #type,
    echo_display,
    #lambda x:x.args,
    partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","carry"]})),
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["sum","carry"]),
    
    get_assignments,
    assign_breakouts, 
    wrap_single_ops_with_breakouts, 
    #cse_assignments, 
    replace_by_nor_not_or, 
    cse_assignments, 
    assign_breakouts,
    #partial(echo_display,label="brent_kung_cla_symboled"),
    #partial(add_cascade_symboled,2)
    partial(brent_kung_cla_symboled,2)
)()#[7].args[1])
#)f_bkcla2 = bkcla2.function_declaration("bkcla2")f_bkcla2
# In[ ]:





# In[57]:


string_to_expression = dict()
string_to_expression["Expression"] = Expression
string_to_expression["Symbol"] = Symbol
string_to_expression["Function"] = Function
string_to_expression["Tuple"] = Tuple
string_to_expression["Dict"] = Dict
string_to_expression["Assignment"] = Assignment
string_to_expression["BreakOut"] = BreakOut
string_to_expression["BooleanExpression"] = BooleanExpression
string_to_expression["BooleanValue"] = BooleanValue
string_to_expression["BooleanAnd"] = BooleanAnd
string_to_expression["BooleanNand"] = BooleanNand
string_to_expression["BooleanNor"] = BooleanNor
string_to_expression["BooleanOr"] = BooleanOr
string_to_expression["BooleanXor"] = BooleanXor
string_to_expression["BooleanNot"] = BooleanNot
string_to_expression["Indexable"] = Indexable
string_to_expression["Indexed"] = Indexed
string_to_expression["ExpressionComponent"] = ExpressionComponent
string_to_expression["MemoryComponent"] = MemoryComponent
string_to_expression["FunctionArgument"] = FunctionArgument
string_to_expression["FunctionDeclaration"] = FunctionDeclaration
string_to_expression["FunctionDefinition"] = FunctionDefinition
string_to_expression["FunctionOutput"] = FunctionOutput
string_to_expression["FunctionCall"] = FunctionCall


# In[58]:


def print_tree(expression, print_result = True, max_item_string_length = 40):
    stack = [(0,expression)]
    lines = []
    single_indent = "  "
    while len(stack) > 0:
        depth, item = stack.pop()
        is_expression = isinstance(item, Expression)
        string = type(item).__name__
        #string += "(" + str(id(type(item) ))+ ")"
        string += ": " 
        #if not is_expression:
        str_item = str(item)
        if len(str_item) > max_item_string_length:
            str_item = str_item[:max_item_string_length][:-3]+"..."
        string += str_item
        prefix = single_indent * depth
        line = prefix + string
        #print(line)
        lines.append(line)
        child_idx = 0
        
        if is_expression:
            for arg in reversed(item.args):
                stack.append((depth+1,arg))
        elif type(item) != str:
            try:
                args = list(iter(item))
                for arg in reversed(args):
                    stack.append((depth+1,arg))
            except TypeError:
                pass

    result = "\n".join(lines)
    if print_result:
        print(result)
    return result


# In[59]:


def expr(argument):
    recurse = expr
    result = lambda x:x
    
    if type(argument) == bool:
        return result(BooleanValue(argument))
    
    elif type(argument) == tuple:
        return result(Tuple(*argument))
    
    return result(map_collection_or_pass(recurse, argument))


# In[60]:


def symbols(string):
    return Tuple(*map(Symbol,string.split(" ")))


# In[61]:


def numbered_symbols(string="x", start_idx=0, count=None, interval=1, infix=""):
    k = 0
    while count is None or k < count:
        idx = start_idx + interval * k
        yield Symbol(string + infix + str(idx))
        k += 1


# In[62]:


def iter_symbols(expression):
    return iter_unique(filter(lambda expr: type(expr) == Symbol, preorder_traversal( expression )))


# In[63]:


def list_symbols(expression, n=None):
    if n is None:
        return list(iter_symbols(expression))
    else:
        return list_n(iter_symbols(expression), n)


# In[64]:


def list_symbols(expression, n=None):
    return list_symbols(expression, n)


# In[65]:


def unused_symbols(expression=None, symbols=None):
    if type(symbols) == str: symbols = numbered_symbols(symbols)
    if symbols is None: symbols = numbered_symbols()
    used_symbols = set(list_symbols(expression)) if expression is not None else set()
    return filter(lambda x:x not in used_symbols, symbols)


# In[66]:


def split_name_id(string, default_id=0, infix="_", to_int=True):
    string = str(string)
    match = re.match(r"^([^\d]*)(|\d+)$", string)
    if match is None: return string, default_id
    int_or = lambda s, default_value: (
        s      if not to_int else
        int(s) if len(s) > 0 else
        default_value
    )
    name = match.group(1)
    id = int_or(match.group(2), default_id)
    if name.endswith(infix): name = name[:-len(infix)]
    return (name, id)


# In[67]:


def function_by_repr(expression):
    symbols = sorted(list_symbols(expression),key=str)
    code = "(lambda {args}: ({expression}))".format(
        args = ", ".join(map(str,symbols)),
        expression = repr(expression)
    )
    #print(code)
    function = eval(code)    
    function.args = symbols
    return function


# In[68]:


def truth_table_by_repr(expression):
    func = function_by_repr(expression)
    num_args = len(func.args)
    result = [
        (args, func(*args))
        for args in itertools.product([False,True], repeat=num_args)
    ]
    return result


# In[69]:


def map_args(func, expression):
    return map(func, iter_args(expression))


# In[70]:


def is_mappable_collection(expression):
    non_collection_types = (str, bytes, bytearray)
    collection_types = (collections.abc.Sequence, dict)
    if isinstance(expression, non_collection_types): return False
    if isinstance(expression, collection_types): return True


# In[71]:


def map_collection(func, expression):
    return type(expression)(map_args(func, expression))


# In[72]:


def map_collection_or_pass(func, expression):
    if is_mappable_collection(expression):
        return map_collection(func, expression)
    else:
        return expression


# In[231]:


def map_expression_args(function, expression, recurse_collection=True):
    # TODO: this should actually be called map_expression_args
    if isinstance(expression, Expression):
        #try:
        return expression.func(*map(function, expression.args))
        #except:
            #print(expression)
            #raise
    
    elif recurse_collection:
        return map_collection_or_pass(function, expression)
    
    else:
        return expression


# In[74]:


def echo(expression,label=None):
    if label is None: print(expression)
    else: print(label, expression)
    return expression


# In[75]:


def echo_display(expression,label=None):
    if label is None: display(expression)
    else: display(label, expression)
    return expression


# In[76]:


def assert_equality_by_table(expr, expression):
    if isinstance(expression, Expression):
        try:
            assert(truth_table_by_repr(expression) == truth_table_by_repr(expr))
        except:
            print(expression)
            print(expr)
            raise
    #print("return ", type(expr), expr)
    return expr


# In[77]:


def evaluate_expr(expression):
    recurse = evaluate_expr
    result = lambda v:v
    result = assert_equality_by_table
    result = lambda v:v
    #arg_is_value = lambda arg: isinstance(arg, (BooleanValue, bool))
    arg_is_value = lambda arg: type(arg) in [BooleanValue, bool]
    def arg_is_value(arg):
        is_value = type(arg) in [BooleanValue, bool]
        #print("is arg a value? " + str(type(arg)) + " " + str(arg))
        #print("is_value", is_value)
        return is_value
    args_are_values = lambda args: all(map(arg_is_value, args))
    get_value = lambda arg: arg if type(arg) == bool else arg.value
    is_true = lambda val: val == True
    is_false = lambda val: val == False
    
    #print("looking at " + str(type(expression)))
    
    if type(expression) == BooleanNot:
        
        assert(len(expression.args) == 1)
        
        arg = recurse(expression.args[0]);
        if arg_is_value(arg):
            return result(BooleanValue(not get_value(arg)))
        else:
            return result(BooleanNot(arg))
                
    elif type(expression) == BooleanOr:
        args = list(map(recurse, expression.args))
        arg_values = [get_value(arg) for arg in args if arg_is_value(arg)]
        args_wo_neutral = list(filter(lambda x: not(arg_is_value(x) and is_false(get_value(x))),args))
        
        if args_are_values(args):
            return result(BooleanValue(any(arg_values)))
        
        elif any(map(is_true,arg_values)):
            return result(BooleanValue(True))
        
        elif len(args) == 1:
            return result(recurse(args[0]))
        
        elif len(args_wo_neutral) < len(args):
            return result(recurse(BooleanOr(*args_wo_neutral)))
        
        else:
            return result(BooleanOr(*args))
    
    elif type(expression) == BooleanAnd:
        args = list(map(recurse, expression.args))
        #print(expression.args)
        #print(args)
        #negated_atom_values = [not get_value(arg) for arg in args if arg_is_value(arg)]
        arg_values = [get_value(arg) for arg in args if arg_is_value(arg)]
        args_wo_neutral = list(filter(lambda x: not(arg_is_value(x) and is_true(get_value(x))),args))
        #print(arg_values)
        
        if args_are_values(args):
            return result(BooleanValue(all(map(is_true,arg_values))))
        
        elif any(map(is_false,arg_values)):
            return result(BooleanValue(False))
        
        elif len(args) == 1:
            return result(recurse(args[0]))
        
        elif len(args_wo_neutral) < len(args):
            return result(recurse(BooleanAnd(*args_wo_neutral)))

        else:
            return result(BooleanAnd(*args))
    
    elif type(expression) == BooleanNand:
        return result(recurse(BooleanNot(BooleanAnd(*expression.args))))
    
    elif type(expression) == BooleanNor:
        return result(recurse(BooleanNot(BooleanOr(*expression.args))))
    
    elif type(expression) == BooleanXor:
        args = list(map(recurse, expression.args))
        
        arg_values = [get_value(arg) for arg in args if arg_is_value(arg)]
        non_value_args = [arg for arg in args if not arg_is_value(arg)]
        
        if len(args) == 0:
            raise ValueError("args are missing")
            
        elif len(args) == 1:
            return result(args[0])
        
        elif len(arg_values) == 0:
            return result(BooleanXor(*non_value_args))
        
        elif len(arg_values) == 1:
            if is_true(arg_values[0]):
                return result(BooleanXor(arg_values[0], *non_value_args))
            else:
                return result(recurse(BooleanXor(*non_value_args)))
            
        elif len(arg_values) > 1:
            evaluated = is_true(arg_values[0])
            for a in arg_values[1:]:
                evaluated ^= is_true(a)
            evaluated = bool(evaluated)
            
            return result(recurse(BooleanXor(evaluated, *non_value_args)))
    
    
    elif type(expression) == BreakOut:
        expr = recurse(expression.expr)
        if arg_is_value(expr):
            return result(BooleanValue(expr))
        else:
            return result(BreakOut(expr))
    
    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))


# In[78]:


evaluate_expr ( BooleanOr(BooleanAnd(Symbol("a"), BooleanValue(False)), BooleanAnd(symbols("b c"))) )


# In[79]:


def replace_by_nand(expression):
    recurse = replace_by_nand
    result = lambda v:v
    #def result(expr):
    #    if isinstance(expression, Expression):
    #        try:
    #            assert(truth_table_by_repr(expression) == truth_table_by_repr(expr))
    #        except:
    #            print(expression)
    #            print(expr)
    #            raise
    #    return expr

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


# In[80]:


def replace_by_nor(expression):
    recurse = replace_by_nor
    result = lambda v:v
    #def result(expr):
    #    if isinstance(expression, Expression):
    #        try:
    #            assert(truth_table_by_repr(expression) == truth_table_by_repr(expr))
    #        except:
    #            print(expression)
    #            print(expr)
    #            raise
    #    return expr

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


# In[81]:


def replace_by_nor_not_or(expression):
    recurse = replace_by_nor
    result = lambda v:v
    #def result(expr):
    #    if isinstance(expression, Expression):
    #        try:
    #            assert(truth_table_by_repr(expression) == truth_table_by_repr(expr))
    #        except:
    #            print(expression)
    #            print(expr)
    #            raise
    #    return expr

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


# In[82]:


def expr_to_sympy(expression):
    recurse = expr_to_sympy
    result = lambda v:v
    
    if type(expression) == Symbol:
        return result(sp.Symbol(expression.name))
    
    if type(expression) == Tuple:
        return result(sp.Tuple(*map(expr_to_sympy,expression.args)))
    
    if type(expression) == Assignment:
        return result(ast.Assignment(*map(expr_to_sympy,expression.args)))
    
    elif type(expression) == BooleanValue:
        return result(sp.true if expression.value == True else sp.false)
    
    # todo: 
    # FunctionArgument
    # FunctionDefinition (ast.FunctionPrototype)
    # FunctionDeclaration (ast.FunctionDefinition)
    # FunctionCall
    
    
    elif isinstance(expression, Expression):
        return result(sp.Function(expression.func.__name__)(*map(recurse, expression.args)))
    
    else:
        return result(map_collection_or_pass(recurse, expression))


# In[ ]:





# In[83]:


def sympy_to_expr(expression):
    recurse = sympy_to_expr
    result = lambda v:v
    #print(expression, type(expression))
    if type(expression) == sp.Symbol:
        return result(Symbol(expression.name))
    
    if type(expression) == sp.Tuple:
        return result(Tuple(*map(recurse,expression.args)))
    
    if type(expression) == ast.Assignment:
        return result(Assignment(*map(recurse,expression.args)))
    
    elif type(expression) == sp.logic.boolalg.BooleanTrue:
        return result(BooleanValue(True))
        
    elif type(expression) == sp.logic.boolalg.BooleanFalse:
        return result(BooleanValue(False))
        
    elif isinstance(expression, sp.core.function.AppliedUndef):
        global string_to_expression
        if expression.name in string_to_expression:
            return result(string_to_expression[expression.name](*map(recurse, expression.args)))
        else:
            return result(Function(expression.name, *map(recurse, expression.args)))
        return result(sp.Function(expression.func.__name__)(*map(recurse, expression.args)))
    
    # todo: 
    # FunctionArgument
    # FunctionDefinition (ast.FunctionPrototype)
    # FunctionDeclaration (ast.FunctionDefinition)
    # FunctionCall

    
    elif isinstance(expression, Expression):
        return result(expression.func(*map(recurse,expression.args)))
    
    elif isinstance(expression, list):
        return result(Tuple(*map(recurse,expression)))

    else:
        return result(map_collection_or_pass(recurse, expression))


# In[84]:


def cse(expression, symbols=None):
    symbols = unused_symbols(expression,symbols)
    symbols = map(expr_to_sympy, symbols)
    return sympy_to_expr(sp.cse(expr_to_sympy(expression),symbols=symbols))


# In[85]:


def identity(x, *args, **kwargs):
    return x


# In[86]:


def replace(expression, needle, replacement, pre_recurse=identity, post_recurse=identity, result=identity, *args, **kwargs):
    def recurse(expr): 
        return post_recurse(
            replace(
                pre_recurse(expr), 
                needle, 
                *args, 
                replacement=replacement,
                pre_recurse=pre_recurse, 
                post_recurse=post_recurse, 
                result=result, 
                **kwargs
            )
        )
    if type(replacement) == type:
        replacement = lambda expr, recurse: replacement(*map(recurse,expr.args))
        
    if (type(needle) == type) and isinstance(expression, needle):
        return result(replacement(expression, recurse))
    
    elif (type(needle) != type) and callable(needle) and needle(expression):
        return result(replacement(expression, recurse))
    
    elif isinstance(expression, list):
        return result(Tuple(*map(recurse,expression)))

    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))


# In[87]:


def replace_with_context(expression, needle, replacement, context=None, new_context=None, pre_recurse=identity, post_recurse=identity, result=identity, *args, **kwargs):
    if new_context is None: new_context = lambda x,c: x
    def recurse(expr): 
        return post_recurse(
            replace_with_context(
                pre_recurse(expr), 
                needle, 
                *args, 
                context=new_context(expression, context),
                replacement=replacement,
                pre_recurse=pre_recurse, 
                post_recurse=post_recurse, 
                result=result, 
                **kwargs
            )
        )
    
    if type(replacement) == type:
        replacement = lambda expr, recurse, context: replacement(*map(recurse,expr.args))
        
    if (type(needle) == type) and isinstance(expression, needle):
        return result(replacement(expression, recurse, context))
    
    elif (type(needle) != type) and callable(needle) and needle(expression, context):
        return result(replacement(expression, recurse, context))
    
    elif isinstance(expression, list):
        return result(Tuple(*map(recurse,expression)))

    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))


# In[ ]:





# In[ ]:





# In[88]:


def generate_breakout_assignments(expression, symbols = None):
    #assert(type(assignments) == Tuple)
    #assert(all(map(lambda item: type(item) == Assignment), assignments))
    
    recurse = generate_breakout_assignments
    result = identity
    
    symbols = unused_symbols(expression, symbols)
    
    if isinstance(expression, BreakOut):
        symbol = next(symbols)
        yield result(Assignment(symbol, expression.expr))
        yield from recurse(expression.expr, symbols)
        
    elif isinstance(expression, Expression):
        for arg in expression.args:
            yield from recurse(arg, symbols)
    
    elif is_mappable_collection(expression):
        for item in expression:
            yield from recurse(item, symbols)


# In[89]:


def assign_breakouts(expression, symbols = None):
    symbols = unused_symbols(expression, symbols)

    result = identity
    
    name_hint_symbols = dict()
    
    def next_symbol(name_hint):
        if name_hint == "":
            return next(symbols)
        
        if name_hint not in name_hint_symbols:
            name_hint_symbols[name_hint] = unused_symbols(expression, name_hint)
        
        return next(name_hint_symbols[name_hint])
    
    # reuse already assigned expressions
    assigned = dict()
    def make_assignment(breakout, recurse):
        if breakout not in assigned:
            lhs = next_symbol(breakout.name_hint)
            rhs = recurse(breakout.expr)
            assigned[breakout] = result(Assignment(lhs, rhs))
        return assigned[breakout]
    
    return replace(
        expression,
        BreakOut,
        make_assignment 
    )


# In[90]:


def replace_tuple_by_assignments(expression, symbols=None):
    if symbols is None: symbols = numbered_symbols()
    symbols = unused_symbols(expression, symbols)
    return replace(
        expression,
        Tuple,
        lambda expr, recurse: Tuple(
            *itertools.starmap(
                Assignment, 
                zip(symbols, map(recurse,expr.args))
            )
        )
    )


# In[91]:


def replace_assignments_by_lhs(expression):
    return replace(
        expression, 
        Assignment, 
        lambda expr, recurse: expr.lhs
    )


# In[92]:


def get_assignments(expression, traversal = preorder_traversal, only_unique=True, sort_key=str):
    is_assignment = lambda x:type(x)==Assignment
    assignments = filter(is_assignment,traversal(expression))
    replace_by_lhs = lambda expression: replace(
        expression, 
        Assignment, 
        lambda expr, recurse: expr.lhs
    )
    replace_rhs = lambda assignment: Assignment(assignment.lhs, replace_by_lhs(assignment.rhs))
    
    assignments = map(replace_rhs, assignments)
    if only_unique:
        assignments = iter_unique(assignments, key=hash)
    assignments = sorted(assignments, key=sort_key)
    assignments = Tuple(*assignments)
    
    #flat_assignments = list(itertools.chain(*map(lambda x:zip_expression(x.lhs, x.rhs, key=is_lvalue), assignments)))

    return assignments


# In[93]:


def flatten_assignments(expression):
    zip_assignment = lambda x: zip_expression(x.lhs, x.rhs, key=is_lvalue, flat=True, as_list=False)
    flat_assignment = lambda x: list(itertools.starmap(Assignment, zip_assignment(x)))
    wrap_tuple = lambda x: x if len(x) == 1 else Tuple(*x)
    return replace(
        expression,
        Assignment,
        lambda x,r: wrap_tuple(flat_assignment(map_expression_args(r,x)))
    )
    
    


# In[94]:


def cse_assignments(assignments, *args, **kwargs):
    new_assignment_tuples, replaced_assignments = cse(assignments, *args, **kwargs)
    new_assignments = (Assignment(lhs,rhs) for lhs,rhs in new_assignment_tuples)
    return Tuple(*new_assignments, *replaced_assignments[0])

def topo_group(expression, sort_in_group = True):
    assignments = get_assignments(expression, preorder_traversal)
    is_lvalue = lambda x: isinstance(x, (Symbol, Indexed, Indexable))
    list_lvalues = lambda x: list(filter(is_lvalue,preorder_traversal(x)))
    get_lhs = lambda x: x.lhs
    is_assign_symbol = lambda symbol: symbol in all_lhs
    no_assign_symbol = lambda x: not any(map(is_assign_symbol, list_lvalues(x)))
    is_independent_assignment = lambda x:no_assign_symbol(x.rhs)
    
    not_assigned = lambda x: x.lhs not in assigned_lhs
    is_symbol_assigned = lambda symbol: (symbol in assigned_lhs) or (symbol not in all_lhs)
    all_symbols_assigned = lambda x: all(map(is_symbol_assigned,list_lvalues(x)))
    rhs_all_symbols_assigned = lambda x: all_symbols_assigned(x.rhs)

    groups = []
    assigned_lhs = set()
    all_lhs = list(map(get_lhs,assignments))

    unassigned = assignments
    groups.append(Tuple(*filter(is_independent_assignment,unassigned)))
    assigned_lhs.update( set(map(get_lhs,groups[-1])) )
    
    unassigned = list(filter(not_assigned, unassigned))
    while len(unassigned) > 0:
        groups.append(Tuple(*filter(rhs_all_symbols_assigned,unassigned)))
        assigned_lhs.update( set(map(get_lhs,groups[-1])) )
        unassigned = list(filter(not_assigned, unassigned))
    
    if sort_in_group:
        
        sort_key = lambda lhs: lhs.args if type(lhs) == Indexed else split_name_id(str(lhs), -1) 
        sort_group = lambda group: sorted(group, key=compose(sort_key,get_lhs))
        groups = list(map(sort_group, groups))
    return groups
# In[95]:


list(map(list,itertools.tee(iter(range(10)),2)))


# In[96]:


list_partition(lambda x:x%2==0, range(10))


# In[97]:


def iter_generic_topo_group(iterable, get_dependees, get_dependencies):
    items = list(iterable)

    get_all_f = lambda f: lambda x: list(itertools.chain(*map(f, x)))
    get_all_dependees = get_all_f(get_dependees)
    get_all_dependencies = get_all_f(get_dependencies)

    is_resolved = lambda x: (x in resolved_dependees) or (x not in all_dependees)
    all_resolved = lambda x: all(map(is_resolved, x))
    all_dependencies_resolved = lambda x: all_resolved(get_dependencies(x))
    
    all_dependees = get_all_dependees(items)
    all_dependencies = get_all_dependencies(items)

    resolved_dependees = set()
    
    unresolved_items = items
    
    while len(unresolved_items) > 0:
        #print(unresolved_items)
        group, unresolved_items = list_partition(all_dependencies_resolved, unresolved_items)
        yield group
        resolved_dependees.update(set(get_all_dependees(group)))        


# In[98]:


def list_generic_topo_group(iterable, get_dependees, get_dependencies):
    return list(iter_generic_topo_group(iterable, get_dependees, get_dependencies))


# In[99]:


all([])


# In[100]:


list_generic_topo_group([(3,2),(2,1,0),(1,0)], lambda x:x[:1], lambda x:x[1:])


# In[101]:


def iter_dependees(x,in_lhs=False):
    #TODO
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

    is_lvalue = lambda x: isinstance(x, (Symbol, Indexed, Indexable))
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


# In[102]:


list(iter_dependees(FunctionDeclaration("f",(FunctionArgument.In(Symbol("a")),FunctionArgument.Out(Symbol("b")))).call(*symbols("x y"))))


# In[103]:


type(None)


# In[104]:


def iter_dependencies(x,parent=None,in_rhs=True):
    #TODO
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
    is_lvalue = lambda x: isinstance(x, (Symbol, Indexed, Indexable))
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


# In[105]:


list(iter_dependees(echo_display(list(FunctionDeclaration("f",(FunctionArgument.In(Symbol("a")),FunctionArgument.Out(Symbol("b")))).call(*symbols("x"))))))


# In[106]:


list(iter_dependencies(Assignment(Symbol("_"), BooleanAnd(*symbols("a b")))))


# In[107]:


def is_lvalue(x):
    return isinstance(x, (Symbol, Indexed, Indexable))


# In[108]:


def iter_lvalues(x, traversal=preorder_traversal):
    return filter(is_lvalue,traversal(x))


# In[109]:


def list_lvalues(x, traversal=preorder_traversal):
    return list(iter_lvalues(x))


# In[110]:


def topo_group(expression, sort_in_group = True):
    assignments = get_assignments(expression, preorder_traversal)
    #is_lvalue = lambda x: isinstance(x, (Symbol, Indexed, Indexable))
    #list_lvalues = lambda x: list(filter(is_lvalue,preorder_traversal(x)))
    get_lhs = lambda x: x.lhs
    get_rhs = lambda x: x.rhs
    get_dependees = compose(
        list,
        iter_unique,
        iter_dependees,
    )
    get_dependencies = compose(
        list,
        iter_unique,
        iter_dependencies,
    )
    
    groups = list_generic_topo_group(
        assignments, 
        get_dependees, 
        get_dependencies
    )

    if sort_in_group:
        
        sort_key = lambda lhs: (
            tuple([lhs.args]) 
            if type(lhs) == Indexed else 
            tuple([split_name_id(str(lhs), -1)])
        )
        sort_group = lambda group: sorted(group, key=compose(
            sort_key,
            list_lvalues,
            get_lhs
        ))
        groups = map(sort_group, groups)
    
    
    groups = Tuple(*itertools.starmap(Tuple,groups))
        
        
    return groups


# In[111]:


(('a',1),) < (('b',1),('b',1))

#print_tree(partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","cout"]}))(
#print_tree(
#partial(print_tree,max_item_string_length=100)(
list(map(list,
topo_group(
#(
    Tuple(
        (Assignment(
            Tuple(Symbol("cout"), *symbols("t_sum0 t_sum1")),
            f_bkcla2.call(*symbols("a0 a1"), *symbols("b0 b1"))
        )),
        (Assignment(
            Tuple(Symbol("_"), *symbols("sum0 sum1")),
            f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
        ))
#         f_bkcla2
#             .call(*symbols("a0 a1"), *symbols("b0 b1"))
#             .assign_to((Symbol("cout"), *symbols("t_sum0 t_sum1")))
#             #.inline(f_bkcla2)
#         ,
#         f_bkcla2
#             .call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
#             .assign_to((Symbol("_"), *symbols("sum0 sum1")))
#             #.inline(f_bkcla2)
#         ,
)
)
)
)
#),True,400)# local variables conflict when inlining just like that, need linker!

#print_tree(partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","cout"]}))(
print_tree(
(
    Tuple(
        Assignment(
            Tuple(Symbol("cout"), *symbols("t_sum0")),
            f_bkcla2.call(*symbols("a0"), *symbols("b0")).inline(f_bkcla2)
        ),
        Assignment(
            Tuple(Symbol("_"), *symbols("sum0")),
            f_bkcla2.call(*symbols("t_sum0"), Symbol("cin")).inline(f_bkcla2)
        )
#         f_bkcla2
#             .call(*symbols("a0 a1"), *symbols("b0 b1"))
#             .assign_to((Symbol("cout"), *symbols("t_sum0 t_sum1")))
#             #.inline(f_bkcla2)
#         ,
#         f_bkcla2
#             .call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
#             .assign_to((Symbol("_"), *symbols("sum0 sum1")))
#             #.inline(f_bkcla2)
#         ,
)
))
#),True,400)topo_group(
    Tuple(
        Assignment(
            Tuple(Symbol("cout"), *symbols("t_sum0 t_sum1")),
            f_bkcla2.call(*symbols("a0 a1"), *symbols("b0 b1"))
        ),
        Assignment(
            Tuple(Symbol("_"), *symbols("sum0 sum1")),
            f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
        )
    )
)topo_group(
    Tuple(
        Assignment(
            Tuple(Symbol("cout"), *symbols("t_sum0 t_sum1")),
            f_bkcla2.call(*symbols("a0 a1"), *symbols("b0 b1"))
        ),
        Assignment(
            Tuple(Symbol("_"), *symbols("sum0 sum1")),
            f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
        )
    )
)
print_tree(partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","cout"]}))(

         f_bkcla2
             .call(*symbols("a0 a1"), *symbols("b0 b1"))
             .assign_to((Symbol("cout"), *symbols("t_sum0 t_sum1")))
             #.inline(f_bkcla2)
         ,
         f_bkcla2
             .call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
             .assign_to((Symbol("_"), *symbols("sum0 sum1")))
             #.inline(f_bkcla2)
         ,
    )
)
#),True,400)def topo_group(expression, sort_in_group = True):
    assignments = get_assignments(expression, preorder_traversal)
    is_lvalue = lambda x: isinstance(x, (Symbol, Indexed, Indexable))
    list_lvalues = lambda x: list(filter(is_lvalue,preorder_traversal(x)))
    get_lhs = lambda x: x.lhs
    
    get_all_lhs_lvalues = lambda x: list(itertools.chain(*map(compose(list_lvalues, get_lhs), x)))
    
    will_be_assigned = lambda x: x in all_lhs_lvalues
    none_will_be_assigned = lambda x: not any(map(will_be_assigned, list_lvalues(x)))
    is_independent_assignment = lambda x:none_will_be_assigned(x.rhs)
    
    not_yet_assigned = lambda x: x not in assigned_lhs_lvalues
    any_of_lhs_not_yet_assigned = lambda x: any(map(not_yet_assigned, list_lvalues(x.lhs)))
    
    is_lvalue_assigned = lambda x: (x in assigned_lhs_lvalues) or (x not in all_lhs_lvalues)
    all_lvalues_assigned = lambda x: all(map(is_lvalue_assigned, list_lvalues(x)))
    rhs_all_lvalues_assigned = lambda x: all_lvalues_assigned(x.rhs)
    
    groups = []
    assigned_lhs_lvalues = set()
    all_lhs_lvalues = get_all_lhs_lvalues(assignments)
    
    unassigned = assignments
    
    groups.append(Tuple(*filter(is_independent_assignment, unassigned)))
    assigned_lhs_lvalues.update(set(get_all_lhs_lvalues(groups[-1])))
    unassigned = list(filter(any_of_lhs_not_yet_assigned, unassigned))
                                
    while len(unassigned) > 0:
        groups.append(Tuple(*filter(rhs_all_lvalues_assigned, unassigned)))
        assigned_lhs_lvalues.update(set(get_all_lhs_lvalues(groups[-1])))
        unassigned = list(filter(any_of_lhs_not_yet_assigned, unassigned))
    
    if sort_in_group:
        
        sort_key = lambda lhs: (
            tuple(map(sort_key,list_lvalues(lhs))) if type(lhs) == Tuple else
            lhs.args if type(lhs) == Indexed else 
            split_name_id(str(lhs), -1) 
        )
        sort_group = lambda group: sorted(group, key=compose(sort_key,get_lhs))
        groups = list(map(sort_group, groups))
    return groups
# In[112]:


topo_group([
    Assignment(symbols("x"), symbols("y z")),
    Assignment(symbols("i j"), symbols("x")),
    Assignment(symbols("a b"), symbols("i j")),
    Assignment(symbols("o"), symbols("a")),
    Assignment(symbols("u"), symbols("b")),
])


# In[ ]:





# In[113]:


def find_assignment_depth(expression):
    return len(topo_group(expression))


# In[114]:


def topo_sort(assignments):
    groups = topo_group(assignments, sort_in_group=True)
    return Tuple(*itertools.chain(*groups))


# In[115]:


def replace_with_breakouts(expression, predicate, *args, **kwargs):
    return replace(expression, predicate, lambda x,r:BreakOut(r(x)), *args, **kwargs)


# In[116]:


def wrap_single_ops_with_breakouts(expression, name_hint="o", parent=None): 
    op_weight_no_symbol = dict(zip(map(id, [Symbol,BreakOut,Assignment]),itertools.repeat(0)))
    skip_breakout = lambda x:(id(type(x)) in set(map(id,[BreakOut,Assignment])))
    include_breakout = lambda x: ( not skip_breakout(x) and isinstance(x,BooleanExpression) )
    #is_single_op = lambda x:(x.count_ops(op_weight_no_symbol) == 1)
    #is_single_op = lambda x:(isinstance(x,Expression) and (x.count_ops(op_weight_no_symbol) > 0))
    
    recurse = lambda x: wrap_single_ops_with_breakouts(x, name_hint, parent=expression)
    result = identity
    if include_breakout(expression) and (not skip_breakout(parent) or parent is None):
        breakout = BreakOut(map_expression_args(recurse, expression), name_hint)
        return result(breakout)
        
    else: 
        return result(map_expression_args(recurse, expression, recurse_collection=True))


# In[ ]:





# In[ ]:





# In[ ]:





# In[117]:


def half_adder(a, b):
    half_sum = BooleanXor(a, b)
    carry = BooleanAnd(a, b)
    return Tuple(half_sum, carry)


# In[118]:


def full_adder(a, b, c):
    sum_ab, carry_ab = half_adder(a, b)
    sum_abc, carry_abc = half_adder(sum_ab, c)
    
    return Tuple(sum_abc, carry_ab | carry_abc)


# In[119]:


def add_cascade(a, b, carry_in):
    carry = carry_in
    result = []
    for a_bit, b_bit in zip(a,b):
        sum_bit, carry = full_adder(a_bit, b_bit, carry)
        #carry = BreakOut(carry)
        result.append(sum_bit)
    return Tuple(Tuple(*map(lambda x:BreakOut(x,"sum"),result)), BreakOut(carry,"carry_out"))


# In[120]:


def add_cascade_symboled(n,a="a",b="b"):
    return add_cascade(list(numbered_symbols(a,0,n)), list(numbered_symbols(b,0,n)), Symbol("carry"))


# In[121]:


def brent_kung_cla(a, b):
    # https://en.wikipedia.org/wiki/Brent%E2%80%93Kung_adder#Basic_model_outline
    # carry_in is always zero
    
    op = lambda a1,b1,a2,b2: Tuple(a1 | (b1 & a2), b1 & b2)
    to_gp = lambda a,b: Tuple(a&b,a^b)

    initial_idcs = lambda num: (list(range(num)), list(range(1,num)))
    advance_idcs = lambda idcs: (idcs[0][1::2], idcs[0][::2])
    to_op = lambda k0,k1: op(*GP_dict[k0][1],*GP_dict[k1][1])
    union_of_idcs = lambda k0,k1: set.union(GP_dict[k0][0],GP_dict[k1][0])

    ab = list(zip(a,b))
    gp = [*itertools.starmap(to_gp,ab)]
    num = len(ab)

    # add breakouts
    gp = [(g,BreakOut(p,"p",expose=False)) for g,p in gp]

    # build tree 
    idcs = initial_idcs(num)
    # initialize GP with gp
    GP_dict = dict()
    GP_dict.update(zip(idcs[0], itertools.starmap(lambda k,x:(set([k]),x),(list(enumerate(gp))))))
    while len(idcs[0]) > 0 and len(idcs[1]) > 0:
        idcs = advance_idcs(idcs)
        GP_dict.update(dict(zip(idcs[0], itertools.starmap(lambda k0,k1:(union_of_idcs(k0,k1),to_op(k0,k1)), zip(*idcs)))))

    # compute full GP[k] for each k
    kset = set()
    k = 0
    kset.add(k)
    while k < num:
        # GP[k] should see all in kset
        unseen_k = set.difference(kset, GP_dict[k][0])
        if len(unseen_k) > 0:
            # connect to last unseen k. this has seen all previous k (by induction)
            idx = max(unseen_k)
            GP_dict[k] = (union_of_idcs(k, idx), to_op(k, idx))

        k += 1
        kset.add(k)

    # collect the actual GP pairs 
    GP = [GP_dict[k][1] for k in range(num)]
    
    # add breakouts
    GP = [(BreakOut(G,"G",expose=False),P) for G,P in GP]

    # compute sum bits by xoring carry(==G[k-1]) with p[k]
    s = [
        # carry_in is zero by definition
        gp[k][1] if k == 0 else (gp[k][1] ^ GP[k-1][0]) 
        for k in range(num)
    ]
    carry_out = GP[num-1][0]

    # add breakouts
    s = [BreakOut(x,"sum") for x in s]
    carry_out = BreakOut(carry_out,"carry_out")
        
    output = Tuple(*s, carry_out)
    
    return output


# In[122]:


def brent_kung_cla_symboled(n,a="a",b="b"):
    return brent_kung_cla(list(numbered_symbols(a,0,n)), list(numbered_symbols(b,0,n)))


# In[123]:


add_cascade_symboled(1)


# In[124]:


list(cse_assignments(get_assignments(assign_breakouts(add_cascade_symboled(8)))))


# In[125]:


for group in topo_group(cse_assignments((get_assignments(assign_breakouts(add_cascade_symboled(8)))))):
    print(group)


# In[ ]:





# In[126]:


list(get_assignments(assign_breakouts(brent_kung_cla_symboled(4))))


# In[127]:


(list(topo_group(assign_breakouts(brent_kung_cla_symboled(8)))))


# In[128]:


(list(topo_group(cse_assignments(assign_breakouts(brent_kung_cla_symboled(8))))))


# In[129]:


op_weight_no_symbol = dict(zip(map(id, [Symbol,BreakOut,Assignment]),itertools.repeat(0)))


# In[130]:


Symbol("a").count_ops(op_weight_no_symbol)


# In[131]:


(list(topo_group(cse_assignments(assign_breakouts(brent_kung_cla_symboled(8))))))[0][0]


# In[132]:


(list(topo_group(cse_assignments(assign_breakouts(brent_kung_cla_symboled(8))))))[0][0].count_ops(op_weight_no_symbol)


# In[133]:


#len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(assign_breakouts(add_cascade_symboled(32))))))))


# In[134]:


len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(assign_breakouts(brent_kung_cla_symboled(8))))))))


# In[135]:


len(list(topo_group(cse_assignments(assign_breakouts(brent_kung_cla_symboled(32))))))

len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nand(cse_assignments(assign_breakouts(add_cascade_symboled(32))))))))))len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nor(cse_assignments(assign_breakouts(add_cascade_symboled(32))))))))))len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nor_not_or(cse_assignments(assign_breakouts(add_cascade_symboled(32))))))))))len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nand(cse_assignments(assign_breakouts(brent_kung_cla_symboled(64))))))))))len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nor(cse_assignments(assign_breakouts(brent_kung_cla_symboled(64))))))))))len(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nor_not_or(cse_assignments(assign_breakouts(brent_kung_cla_symboled(64))))))))))
# In[136]:


(list(topo_group(assign_breakouts(wrap_single_ops_with_breakouts(cse_assignments(replace_by_nor_not_or(cse_assignments(assign_breakouts(brent_kung_cla_symboled(4))))))))))


# In[137]:


decide_predicate_usage([Symbol,(lambda x:x.name=="f")], True, all)(BooleanValue(True))


# In[ ]:





# In[138]:


def resolve_noop_assignments(expression, excluded=None, keep=None):
    is_excluded           = decide_predicate_usage(excluded, False, any)
    must_keep             = decide_predicate_usage(keep, False, any)
    is_not_excluded       = lambda x: not is_excluded(x)
    must_not_keep         = lambda x: not must_keep(x)
    is_assignment         = lambda x: type(x) == Assignment
    assigns_from_symbol   = lambda x: type(x.rhs) == Symbol
    any_arg_not_excluded  = lambda x: any(map(is_not_excluded,x.args))
    any_arg_must_not_keep = lambda x: any(map(is_not_excluded,x.args))
    
    assignments = topo_sort(expression)
    
    sorted_lhs = list(map(lambda x:x.lhs, assignments))
    
    # find assignments which just assign from another symbol
    noop_assignments = list(filter(
        #lambda x: is_assignment(x) and assigns_from_symbol(x) and any_arg_not_excluded(x),
        lambda x: assigns_from_symbol(x) and any_arg_not_excluded(x),
        assignments
    ))
    
    pairs = list(filter(
        lambda x:(x.lhs,x.rhs), #if is_excluded(x.lhs) else (x.rhs,x.lhs)),
        noop_assignments
    ))
    
    # transitive hulls of assignments between symbols
    # i.e. which groups of symbols are assigned the same expr
    groups = []
    groups_dict = dict()
    for k, (lhs,rhs) in enumerate(pairs):
        lhs_in_group = lhs in groups_dict
        rhs_in_group = rhs in groups_dict
        lhs_not_in_group = lhs not in groups_dict
        rhs_not_in_group = rhs not in groups_dict
        
        if lhs_not_in_group and rhs_not_in_group:
            group = set([lhs,rhs])
            k = len(groups)
            groups.append(group)
            groups_dict[lhs] = k
            groups_dict[rhs] = k
            
        elif lhs_in_group and  rhs_in_group:
            assert(groups_dict[lhs] == groups_dict[rhs])
            
        elif lhs_in_group or rhs_in_group:
            k = groups_dict[lhs if lhs_in_group else rhs]
            group = groups[k]
            group.add(lhs)
            group.add(rhs)
    
    # for each group find symbol which will be assigned the actual expression.
    source_symbols = list(map(
        compose(
            #next, iter, # take first
            lambda x: (None if len(x) == 0 else compose(next,iter)(x)),
            partial(sorted,key=lambda x:(not(is_excluded(x) or must_keep(x)),sorted_lhs.index(x))),
            list,
            #partial(filter,lambda x:is_not_excluded(x) and must_not_keep(x)),
            #partial(filter,must_not_keep),
            
        ),groups
    ))
    
    # find symbols in groups that will be replaced by their source_symbol
    replaced_symbols = list(map(
        compose(
            #next, iter, # take first
            list,
            partial(sorted,key=str),
            list,
            partial(filter,lambda x:is_not_excluded(x) and must_not_keep(x) and x not in source_symbols),
            #partial(filter,must_not_keep),
            
        ),groups
    ))
    
    # build replacement pairs
    replacements = dict([
        (replace, by)
        for replace_list, by in zip(replaced_symbols, source_symbols)
        if by is not None
        for replace in replace_list
    ])
    
    #print("sorted_lhs", sorted_lhs)
    #print("noop_assignments", noop_assignments)
    #print("pairs", pairs)
    #print("groups", groups)
    #print("source_symbols", source_symbols)
    #print("replaced_symbols", replaced_symbols)
    #print("replacements", replacements)
    
    # apply replacements
    replaced = replace(
        expression,
        Symbol,
        lambda x,r: x if x not in replacements else replacements[x]
    )
    # discard assignments of form x=x
    replaced = list(filter(
        lambda x: not (x.lhs == x.rhs),
        replaced
    ))
    return replaced


# In[139]:


#type
compose(
    #list,
    #type,
    #lambda x:x[7],
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["G","carry","sum"]),
    
    get_assignments,
    assign_breakouts, 
    wrap_single_ops_with_breakouts, 
    cse_assignments, 
    replace_by_nor_not_or, 
    cse_assignments, 
    assign_breakouts,
    partial(brent_kung_cla_symboled,1)
)()#[7].args[1])
#)


# In[ ]:





# In[140]:


#type
compose(
    #len,
    #list,
    #type,
    #lambda x:x[7],
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["sum","carry"]),
    
    get_assignments,
    assign_breakouts, 
    wrap_single_ops_with_breakouts, 
    #cse_assignments, 
    replace_by_nor_not_or, 
    cse_assignments, 
    assign_breakouts,
    #partial(add_cascade_symboled,2)
    partial(brent_kung_cla_symboled,2)
)()#[7].args[1])
#)


# In[141]:


@dataclass
class ComponentConfig:
    sections   : typing.List[ComponentSection]         = dataclasses.field(default_factory=lambda:None) # use default
    predicates : typing.Dict[ComponentSection, object] = dataclasses.field(default_factory=lambda:{})   # use default
    # default: preserve assignment topological preorder traversal
    symbol_order  : typing.Dict[ComponentSection, object] = dataclasses.field(default_factory=lambda:{})   
    
    def update(self, other):
        if other.sections is not None: 
            self.sections = other.sections 
        self.predicates.update(other.predicates)
        self.symbol_order.update(other.symbol_order)

    def default_config(updates=None):
        cfg = ComponentConfig(
            sections = [
                ComponentSection.Input,
                ComponentSection.Output,
                ComponentSection.Local,
                ComponentSection.Data,
                ComponentSection.Rest,
            ],
            predicates = {},
            symbol_order = {
                ComponentSection.Input:  split_name_id,
                ComponentSection.Output: split_name_id,
            }
        )
        if updates is not None:
            cfg.update(updates)
        return cfg


# In[142]:


@dataclass
class MemoryComponentConfig:
    # default: preserve existing order 
    symbol_order  : typing.Dict[ComponentSection, object] = dataclasses.field(default_factory=lambda:{})   
    
    def update(self, other):
        self.symbol_order.update(other.symbol_order)

    def default_config(updates=None):
        cfg = MemoryComponentConfig(
            symbol_order = {
                ComponentSection.Input:  split_name_id,
                ComponentSection.Output: split_name_id,
            }
        )
        if updates is not None:
            cfg.update(updates)
        return cfg


# In[143]:


ComponentConfig()


# In[144]:


list(iter(ComponentSection))


# In[145]:


def make_expression_component(expression, cfg:ComponentConfig=None):
    if cfg is None: cfg = ComponentConfig()
   
    cfg = ComponentConfig.default_config(updates=cfg)
    
    expression_without_nested_local = replace(
        expression,
        ExpressionComponent,
        #lambda x,r: ExpressionComponent({
        #    ComponentSection.Input:  x.section_symbols[ComponentSection.Input],
        #    ComponentSection.Output: x.section_symbols[ComponentSection.Output],
        #}, Tuple())
        lambda x,r: Tuple(*(
            Assignment(output_symbol, Tuple(*x.section_symbols[ComponentSection.Input]))
            for output_symbol in x.section_symbols[ComponentSection.Output]
        ))
    )
    assignments = topo_sort(expression_without_nested_local)
    print(assignments)
    assigned_symbols = [x.lhs for x in assignments] + []
    assigned_symbols_set = set(assigned_symbols)
    
    all_symbols = list_symbols(assignments)
    
    predicates = {}
    no_input_no_output = lambda x: not predicates[ComponentSection.Input](x) and not predicates[ComponentSection.Output](x)
    default_predicates = {
        ComponentSection.Input:  lambda x: x not in assigned_symbols_set, 
        ComponentSection.Output: False, 
        ComponentSection.Local:  no_input_no_output, 
        ComponentSection.Data:   False, 
        ComponentSection.Rest:   False, 
    }
    predicate_or_default = lambda x:cfg.predicates[x] if x in cfg.predicates else default_predicates[x]
    predicates = dict(
        map(
            lambda x: (x, decide_predicate_usage(predicate_or_default(x),default_predicates[x],any)),
            ComponentSection
        )
    )
    section_symbols = dict(
        map(
            lambda x: (x, list(filter(predicates[x],all_symbols))),
            ComponentSection
        )
    )
    
    for section, sort_key in cfg.symbol_order.items():
        if sort_key is not None:
            section_symbols[section]  = list(sorted(section_symbols[section],  key=sort_key))
            
    for key in section_symbols.keys():
        section_symbols[key] = Tuple(*section_symbols[key])
    
    section_symbols = Dict(*section_symbols.items())
    
    return ExpressionComponent(
        section_symbols,
        expression
    )


# In[146]:


#type
bkcla2 = compose(
    #len,
    #list,
    #type,
    echo_display,
    #lambda x:x.expr,
    partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","carry"]})),
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["sum","carry"]),
    
    get_assignments,
    assign_breakouts, 
    
    #wrap_single_ops_with_breakouts, 
    #cse_assignments, 
    #replace_by_nor_not_or, 
    cse_assignments, 
    assign_breakouts,
    #partial(echo_display,label="brent_kung_cla_symboled"),
    #partial(add_cascade_symboled,2)
    partial(brent_kung_cla_symboled,2)
)()#[7].args[1])
#)


# In[147]:


f_bkcla2 = bkcla2.function_definition("bkcla2")


# In[148]:


f_bkcla2.call(*symbols("a0 a1"), *symbols("b0 b1")).assign_to((Symbol("cout"), *symbols("t_sum0 t_sum1")))


# In[149]:


f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False).assign_to((Symbol("_"), *symbols("sum0 sum1")))


# In[150]:


f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False).assign_to((Symbol("_"), *symbols("sum0 sum1"))).kwarguments


# In[151]:


f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False).assign_to((Symbol("_"), *symbols("sum0 sum1"))).inline(f_bkcla2)


# In[152]:


Dict()


# In[153]:


#print_tree(partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","cout"]}))(
#print_tree(
#partial(print_tree,max_item_string_length=100)(
list(map(list,
topo_group(
#(
    Tuple(
        (Assignment(
            Tuple(Symbol("cout"), *symbols("t_sum0 t_sum1")),
            f_bkcla2.call(*symbols("a0 a1"), *symbols("b0 b1"))
        )),
        (Assignment(
            Tuple(Symbol("_"), *symbols("sum0 sum1")),
            f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
        ))
#         f_bkcla2
#             .call(*symbols("a0 a1"), *symbols("b0 b1"))
#             .assign_to((Symbol("cout"), *symbols("t_sum0 t_sum1")))
#             #.inline(f_bkcla2)
#         ,
#         f_bkcla2
#             .call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
#             .assign_to((Symbol("_"), *symbols("sum0 sum1")))
#             #.inline(f_bkcla2)
#         ,
)
)
)
)
#),True,400)


# In[154]:


compose(
    
    #evaluate_expr,
    #lambda x:x.expr[-2].rhs,
    evaluate_expr,
    f_bkcla2.inline
)(
    Assignment(
        Tuple(Symbol("_"), *symbols("sum0 sum1")),
        f_bkcla2.call(*symbols("t_sum0 t_sum1"), Symbol("cin"), False)
    )
)


# In[155]:


compose(
    
    #evaluate_expr,
    #lambda x:x.expr[-2].rhs,
    #type,
    #print_tree,
    #lambda x:x[3].rhs,
    #partial(resolve_noop_assignments,keep="sum"),
    lambda x:x.expr,
    #evaluate_expr,
    evaluate_expr,
    f_bkcla2.inline
)(
    Assignment(
        Tuple(Symbol("_"), *symbols("sum0 sum1")),
        f_bkcla2.call(*symbols("t_sum0 t_sum1"), False, False)
    )
)


# In[267]:


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


# In[287]:


def replace_assignments_by_breakouts(expression, exclude=None, include=None):
    is_excluded = decide_predicate_usage(exclude, default_predicate=False)
    is_included = decide_predicate_usage(include, default_predicate=True)
        
    assignments = compose(
        list,
        partial(
            filter, 
            lambda x: (
                any(map(is_included,list_lvalues(x.lhs))) 
                and not 
                all(map(is_excluded,list_lvalues(x.lhs))) 
            )
        ),
        topo_sort,
        get_assignments,
        flatten_assignments,
        
    )(expression)
    #flat_assignments = list(itertools.chain(*map(lambda x:zip_expression(x.lhs, x.rhs, key=is_lvalue), assignments)))
    #assignments = list(itertools.chain(*map(lambda x:zip_expression(x.lhs, x.rhs, key=is_lvalue), assignments)))
    display(assignments)
    #replacements = dict(assignments)
    #replacements = dict([(k,BreakOut(v, k.name)) for k,v in assignments])
    #return replacements
    replaced = expression
#     for k,(lhs,rhs) in enumerate(assignments):
#         # replace lhs in expression by breakout(rhs,lhs)
#         count_replacements = itertools.count()
#         replaced = replace(
#             replaced,
#             lambda x: x == lhs,
# #            lambda x,r: rhs,
#             lambda x,r: BreakOut(rhs,name_hint=str(x)) if next(count_replacements)>=0 and True else None,
# #             #count_replacements,
# #             #lambda x,c: c
#         )
#         #display("lhs")
#         #display(print_tree(lhs))
#         #display("lhs,rhs")
#         #display(lhs,rhs)
#         #display("replaced")
#         #display(replaced)
#         #display("count_replacements", count_replacements)
#         count_replacements = next(count_replacements)
#         # if any replacement occured, replace assignment by breakout
#         if count_replacements > 0:
#             pass
    
    #def replace_in_rhs
    
#     replaced_wo_assignments = replace_with_context(
#         expression,
#         Assignment,
#         lambda x,r,c: (
#             map_expression_args(r,x)
#         ),
#     )
    #return replaced
    return replacements
    return assignments


# In[288]:


foo = (
    Tuple(
        Assignment(
            Tuple(Symbol("_"), *symbols("sum0 sum1")),
            f_bkcla2.call(*symbols("t_sum0 t_sum1"), False, False)
        ),
        Assignment(
            symbols("t_sum0 t_sum1 c"),
            symbols("a b d")
        )
    )
)


# In[289]:


replacements=compose(
    #lambda x: zip_expression(x[1].lhs, x[1].rhs, key=is_lvalue),
    #list,
    #list,topo_sort,
    replace_assignments_by_breakouts,
    #lambda x:x.expr,
    evaluate_expr,
    echo_display,
    #f_bkcla2.inline,
    #lambda x:Tuple(*x),
    #partial(replace,needle=Assignment,replacement=lambda x,r:f_bkcla2.inline(x) if type(x.rhs) == FunctionCall and x.rhs.function_declaration == f_bkcla2.function_declaration else map_expression_args(r,x))
)(foo)


# In[ ]:


foo, replacements, type(replacements)


# In[216]:


map_expression_args(identity,foo.args[0].args[1])


# In[261]:


used_replacements = Dict()


# In[266]:


replace_lvalues_in_usage_context(foo, replacements, used_replacements=used_replacements)


# In[283]:


replace_lvalues_in_usage_context(foo, dict([(k,BreakOut(v, k.name)) for k,v in replacements.items()]), used_replacements=used_replacements)


# In[264]:


used_replacements


# In[ ]:





# In[276]:


def strip_lvalues_bindings(expression, strip):
    """
    strip out lvalue bindings in Assignment.
    
    e.g.
    strip_lvalue_assignments( Assignment(*symbols("a b c"), *symbols("d e f")), strip={Symbol("a")} )
    > [(b,c) = (e,f)]
    """
    recurse = partial(strip_lvalues_bindings, strip=strip)
    result = identity
    
    def strip_lvalues_bindings_from_assignment(lhs, rhs, r):
        # recursively walk down lhs and rhs in tandem
        return Assignment(lhs,rhs)
    
    return replace(
        expression,
        Assignment,
        lambda x,r: strip_lvalues_bindings_from_assignment(x.lhs, x.rhs, r)
    )
    
#     if is_lvalue(expression) and expression in strip:
#         
#     else:
#         return result(map_expression_args(
#             recurse,
#             expression, 
#             recurse_collection=True)
#         )


# In[277]:


strip_lvalues_bindings( Assignment(symbols("a b c"), symbols("d e f")), strip={Symbol("a")} )


# In[281]:


(flatten_assignments( Assignment(
    symbols("a b c"), 
    FunctionDeclaration(
        "f",
        Tuple(
            *map(
                FunctionArgument.Out,
                symbols("x y z")
            )
        )
    ).call()
) )

    recurse = partial(zip_expression, key=key, flat=flat, as_list=as_list, longest=longest)
    zip_func = itertools.zip_longest if longest else zip
    if any(map(key,args)):
        gen = iter([args])
    elif flat:
        gen = (
            item
            for tpl in zip_func(*map(iterator,args))
            for item in ([tpl] if any(map(key,tpl)) else recurse(*tpl))
        )
    else:
        gen = (
            tpl if any(map(key,tpl)) else recurse(*tpl)
            for tpl in zip_func(*map(iterator,args))
        )
    
    return list(gen) if as_list else gen
# In[ ]:




([(_, sum0, sum1) = bkcla2(t_sum0, t_sum1, False, False, ...)], [(t_sum0, (t_sum1, c)) = (a, (b, d))])
([(_, sum0, sum1) = bkcla2(BreakOut(a,"t_sum0"), BreakOut(b,"t_sum1"), False, False, ...)], [((c,),) = ((d,),)])
# In[1526]:


t_sum0,t_sum1,c,a,b,d=symbols("t_sum0 t_sum1 c a b d")


# In[1527]:


Assignment( Tuple(t_sum0, Tuple(t_sum1, c)) , Tuple(a, Tuple(b, d)) )


# In[1548]:


compose(
    identity,
    #print_tree,
    #lambda x:x[0],
    
)(zip_expression( Tuple(t_sum0, Tuple(t_sum1, c)) , Tuple(a, Tuple(b, d)) , key=decide_predicate_usage(Symbol)))


# In[1543]:


zip_expression(
    (a,b),
    (c,d),
    key=decide_predicate_usage(Symbol)
)


# In[1540]:


zip_expression(
    symbols("b c"),
    FunctionDeclaration("f",[FunctionArgument.In(Symbol("x")),*map(FunctionArgument.Out, symbols("y z"))]).call(Symbol("a")),
    key=decide_predicate_usage(Symbol)
)


# In[1453]:





# In[1271]:


evaluate_expr ( BooleanOr(BooleanAnd(Symbol("a"), BooleanValue(False)), BooleanAnd(symbols("b c"))) )


# In[1266]:


BooleanOr(BooleanAnd(*symbols("a b")))


# In[1200]:


raise NotImplementedError()


# In[ ]:





# In[253]:


bkcla2_add_cin = compose(
    echo_display,
    partial(
        bkcla2,
        *symbols("b0 b1 cin"), BooleanValue(False),
        sum0=Symbol("t_sum0"),
        sum1=Symbol("t_sum1"),
        carry_out0=Symbol("cout"))
)()


# In[254]:


bkcla2_add_cin_sum = echo_display(list(filter(decide_predicate_usage("sum",False),bkcla2_add_cin.section_symbols[ComponentSection.Output])))


# In[250]:


bkcla2_b = list(filter(decide_predicate_usage("b",False),bkcla2.section_symbols[ComponentSection.Input]))


# In[251]:


bkcla2_b


# In[274]:


print_tree(partial(make_expression_component,cfg=ComponentConfig(predicates={ComponentSection.Output:["sum","cout"]}))(
    Tuple(
        bkcla2(
            *symbols("b0 b1 cin"), BooleanValue(False),
            sum0=Symbol("t_sum0"),
            sum1=Symbol("t_sum1"),
            carry_out0=Symbol("cout")
        ), 
        bkcla2(**dict(zip(
            ["b0", "b1"], 
            [Symbol("t_sum0"),Symbol("t_sum1")]
        )))
    )
),True,400)


# In[166]:


r


# In[ ]:


def memory_link_components(expression, address_name: str="mem", addresses=None, cfg:MemoryComponentConfig=None):
    if cfg is None: cfg = MemoryComponentConfig()
    if addresses is None: addresses = itertools.count()
    if type(addresses) == int: addresses = itertools.count(addresses)

    cfg = ComponentConfig.default_config(updates=cfg)

    


# In[ ]:


def make_memory_component(expression, address_name: str="mem", addresses=None, cfg:MemoryComponentConfig=None):
    if cfg is None: cfg = MemoryComponentConfig()
    if addresses is None: addresses = itertools.count()
    if type(addresses) == int: addresses = itertools.count(addresses)

    cfg = ComponentConfig.default_config(updates=cfg)
 
    
    for section, sort_key in cfg.sort_keys.items():
        if sort_key is not None:
            section_symbols[section]  = list(sorted(section_symbols[section],  key=sort_key))

    ordered_symbols = itertools.chain(*map(lambda x:section_symbols[x],cfg.sections))
    
    #ordered_symbols = itertools.chain(input_symbols, assigned_symbols, output_symbols)
    address_map = dict(zip(ordered_symbols, addresses))
    
    return ExpressionComponent(
        section_symbols,
        address_map,
        expression
    )


# In[111]:


def assign_addresses(expression, address_name: str="mem", addresses=None, cfg:AssignAddressesConfig=None):
    
    
    #print(address_map)
    
    replaced = replace(expression, Symbol, lambda x,r:Indexed(address_name, address_map[x]) if x in address_map else x)
    
    return Tuple(
        section_symbols,
        address_map,
        replaced
    )


# In[ ]:





# In[ ]:





# In[115]:


def make_component_list(address_map__expression):
    address_map, expression = address_map__expression
    #print_tree(address_map)
    
    
    index_or_pass = lambda x: x.index if hasattr(x,"index") else x
    replaced = replace(
        expression,
        Assignment,
        lambda x,r: Tuple(
            type(x.rhs),
            index_or_pass(x.lhs),
            *map(index_or_pass,x.rhs.args)
        )
    )
    
    return Tuple(
        address_map,
        replaced
    )


# In[116]:


def group_components_by_type(address_map__components):
    address_map, components = address_map__components
    #print_tree(address_map)
    
    
    groups = defaultdict(lambda:list())
    for component in components:
        func,out,a,b = component
        groups[func].append((out,a,b))
    
    return Tuple(
        address_map,
        dict(groups)
        #Tuple(*itertools.starmap(Tuple,groups.items()))
    )


# In[117]:


left_pad = lambda x, desired_len, padding: type(x)(max(0,desired_len-len(x))*[padding] + list(x))


# In[118]:


right_pad = lambda x, desired_len, padding: type(x)(list(x) + max(0,desired_len-len(x))*[padding])


# In[119]:


to_digits = lambda x: list(map(lambda x:1*int(x),bin(x)[2:][::-1]))


# In[120]:


@numba.jit()
def run_nor_netlist(mem, idx_in, idx_out, netlist):
    num = len(netlist)
    for k in range(num):
        out,a,b = netlist[k]
        mem[idx_out,out] =  ~(mem[idx_in,a] | mem[idx_in,b])
        #mem[idx_out,out] &= 0b1
@numba.jit()
def run_nand_netlist(mem, idx_in, idx_out, netlist):
    num = len(netlist)
    for k in range(num):
        out,a,b = netlist[k]
        mem[idx_out,out] =  ~(mem[idx_in,a] & mem[idx_in,b])
        #mem[idx_out,out] &= 0b1
@numba.jit()
def run_and_netlist(mem, idx_in, idx_out, netlist):
    num = len(netlist)
    for k in range(num):
        out,a,b = netlist[k]
        mem[idx_out,out] =  (mem[idx_in,a] & mem[idx_in,b])
        #mem[idx_out,out] &= 0b1
@numba.jit()
def run_or_netlist(mem, idx_in, idx_out, netlist):
    num = len(netlist)
    for k in range(num):
        out,a,b = netlist[k]
        mem[idx_out,out] =  (mem[idx_in,a] | mem[idx_in,b])
        #mem[idx_out,out] &= 0b1
@numba.jit()
def run_not_netlist(mem, idx_in, idx_out, netlist):
    num = len(netlist)
    for k in range(num):
        out,a,b = netlist[k]
        mem[idx_out,out] =  ~mem[idx_in,a]
        #mem[idx_out,out] &= 0b1
@numba.jit()
def run_xor_netlist(mem, idx_in, idx_out, netlist):
    num = len(netlist)
    for k in range(num):
        out,a,b = netlist[k]
        mem[idx_out,out] =  (mem[idx_in,a] ^ mem[idx_in,b])
        #mem[idx_out,out] &= 0b1        


# In[ ]:





# In[ ]:





# In[ ]:





# In[121]:


#type
#address_map, netlist = 
compose(
    #len,
    
    list,
    #dict,
    #partial(map,list),
    #partial(map,lambda x:type(x.rhs).__name__),
    #type,
    #lambda x:x[7],
    #group_components_by_type,
    #make_component_list,
    #topo_group, 
    #topo_sort,
    #lambda x: [x[0], list(x[1])],
    #partial(assign_addresses,output_predicate=["sum","carry_out"]),
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["sum","carry"]),
    
    get_assignments,
    assign_breakouts, 
    wrap_single_ops_with_breakouts, 
    #cse_assignments, 
    #replace_by_nor_not_or, 
    cse_assignments, 
    assign_breakouts,
    #partial(add_cascade_symboled,2)
    partial(brent_kung_cla_symboled,2)
)()#[7].args[1])
#)


# In[ ]:





# In[ ]:





# In[ ]:





# In[122]:


#type
compose(
    #len,
    
    #list,
    #dict,
    #partial(map,list),
    #partial(map,lambda x:type(x.rhs).__name__),
    #type,
    #lambda x:x[7],
    #group_components_by_type,
    #make_component_list,
    len, 
    topo_group, 
    #topo_sort,
    partial(assign_addresses,address_name="mem",addresses=0,cfg=AssignAddressesConfig(predicates={AddressSection.Output:["sum","carry"]})),
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["sum","carry"]),
    
    get_assignments,
    assign_breakouts, 
    wrap_single_ops_with_breakouts, 
    #cse_assignments, 
    #replace_by_nor_not_or, 
    replace_by_nand, 
    cse_assignments, 
    assign_breakouts,
    #partial(add_cascade_symboled,2)
    partial(brent_kung_cla_symboled,63)
)()#[7].args[1])
#)


# In[106]:


#type
address_map, netlist = compose(
    #len,
    
    #list,
    #dict,
    #partial(map,list),
    #partial(map,lambda x:type(x.rhs).__name__),
    #type,
    #lambda x:x[7],
    group_components_by_type,
    make_component_list,
    #topo_group, 
    #topo_sort,
    partial(assign_addresses,address_name="mem",addresses=0,cfg=AssignAddressesConfig(predicates={AddressSection.Output:["sum","carry"]})),
    topo_sort,
    #topo_group, 
    partial(resolve_noop_assignments,keep=["sum","carry"]),
    
    get_assignments,
    assign_breakouts, 
    wrap_single_ops_with_breakouts, 
    #cse_assignments, 
    #replace_by_nor_not_or, 
    replace_by_nand,
    #replace_by_nor,
    cse_assignments, 
    assign_breakouts,
    #partial(add_cascade_symboled,2)
    partial(brent_kung_cla_symboled,63)
)()#[7].args[1])
#)


# In[107]:


list(address_map.items())[-1]


# In[108]:


netlist


# In[ ]:





# In[109]:


def netlist_to_arr(netlist, dtype=np.uint32):
    max_tuple_len = max(itertools.chain(map(len,netlist),[1]))
    list_len = len(netlist)
    arr = np.zeros(shape=(list_len, max_tuple_len), dtype=dtype)
    for k,tpl in enumerate(netlist):
        arr[k,:len(tpl)] = tpl
    return arr


# In[110]:


def netlist_to_arrs(netlist, dtype=np.uint32):
    result = defaultdict(lambda:netlist_to_arr([], dtype))
    result.update(dict([
        (key,netlist_to_arr(lst, dtype))
        for key, lst in netlist.items()
    ]))
    return result


# In[111]:


def print_mem(symbol_predicate=None):
    
    foo = dict([
        (name, mem[toggle,k])
        for k,name in 
        raddress_map.items()
    ])
    for key, value in foo.items():
        print(str(key) + ": " + str(value))


# In[112]:


netlist_arrs = netlist_to_arrs(netlist)


# In[113]:


netlist_arrs


# In[114]:


#nor_netlist_arr = netlist_to_arr(netlist[BooleanNor])

nor_netlist_arr[:5].Tnor_netlist_arr = np.array([
    (1,0,0),
    (2,1,1),
    (3,2,2),
    (4,3,3),
    (5,4,4),
    (6,5,5),
    (7,6,6),
    (0,7,7),
])
# In[ ]:





# In[115]:


raddress_map = dict(map(reversed,address_map.items()))


# In[116]:


#raddress_map


# In[117]:


(list(raddress_map.values())[0]).name


# In[118]:


mem = np.zeros((2,1536), dtype=np.uint32)


# In[ ]:




mem[:,:2] = right_pad(to_digits(0b01),2,0)
mem[:,2:4] = right_pad(to_digits(0b10),2,0)mem[:,:4] = right_pad(to_digits(3),4,0)
mem[:,4:8] = right_pad(to_digits(10),4,0)
# In[119]:


mem[:,address_map[Symbol("a0")]] = 1
mem[:,address_map[Symbol("a1")]] = 0
mem[:,address_map[Symbol("b0")]] = 0
mem[:,address_map[Symbol("b1")]] = 0


# In[120]:


toggle = 0


# In[121]:


run_nor_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNor])
run_nand_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNand])
run_or_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanOr])
run_and_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanAnd])
run_xor_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanXor])
run_not_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNot])
toggle = 1-toggle
print_mem()


# In[698]:


truth_table = []
mem*=0
for a0,a1,b0,b1 in itertools.product((0,1),(0,1),(0,1),(0,1)):
    mem[:,address_map[Symbol("a0")]] = a0
    mem[:,address_map[Symbol("a1")]] = a1
    mem[:,address_map[Symbol("b0")]] = b0
    mem[:,address_map[Symbol("b1")]] = b1
    for k in range(22): # there where 7 groups in topo_group
        run_nor_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNor])
        run_nand_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNand])
        run_or_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanOr])
        run_and_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanAnd])
        run_xor_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanXor])
        run_not_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNot])
        toggle = 1-toggle
    sum0 = mem[toggle,address_map[Symbol("sum0")]]
    sum1 = mem[toggle,address_map[Symbol("sum1")]]
    carry_out0 = mem[toggle,address_map[Symbol("sum2")]]
    truth_table.append((a0,a1,b0,b1,sum0,sum1,carry_out0))
truth_table = np.array(truth_table)
print(truth_table)
print(truth_table ^true_truth_table)


# In[531]:


#true_truth_table = truth_table


# In[532]:


def bits_to_int(bits, lsb=True):
    if not lsb: bits = list(reversed(bits))
    return sum(itertools.starmap(lambda x,k:(1<<k)*x, zip(bits,itertools.count())))


# In[533]:


def int_to_bits(int_v, lsb=True):
    bits = list(map(lambda x:ord(x)-ord('0'),bin(int_v)[2:]))
    if lsb: bits = list(reversed(bits)) 
    return bits


# In[534]:


def perform_add(a, b, num_bits = 8):
    abits = right_pad(int_to_bits(a), num_bits, 0)
    bbits = right_pad(int_to_bits(b), num_bits, 0)
    global mem
    global toggle
    global address_map
    global netlist_arrs
    
    mem*=0
    for k in range(num_bits):
        mem[:,address_map[Symbol("a%d" % k)]] = abits[k]
        mem[:,address_map[Symbol("b%d" % k)]] = bbits[k]
    
    for k in range(22): # there where 7 groups in topo_group
        run_nor_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNor])
        run_nand_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNand])
        run_or_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanOr])
        run_and_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanAnd])
        run_xor_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanXor])
        run_not_netlist(mem, toggle, 1-toggle, netlist_arrs[BooleanNot])
        toggle = 1-toggle
        
    sumbits = [
        mem[toggle,address_map[Symbol("sum%d" % k)]]
        for k in range(num_bits)
    ]
    carry_out0 = mem[toggle,address_map[Symbol("carry_out0")]]
    sumbits.append(carry_out0)
    sum_value = bits_to_int(sumbits)
    return sum_value#, carry_out0


# In[762]:


@numba.jit
def nb_write_number_bits(number, num_bits, parallel, mem, address):
    #one = (1 << parallel)-1
    #avoid overflow with more complicated formula:
    one = ((1 << (parallel-1))-1)+(1 << (parallel-1))

    for k in range(num_bits):
        mem[:,address+k] = one if (number & (1 << k)) != 0 else 0
    


# In[558]:


@numba.jit
def nb_perform_add(a, b, num_bits, iterations, mem, address_a, address_b, address_sum, address_carry, netlist_nand):
    #mem*=0
    for k in range(num_bits):
        mem[:,address_a+k] = 0b1 if (a & (1 << k)) != 0 else 0
        mem[:,address_b+k] = 0b1 if (b & (1 << k)) != 0 else 0
    
    toggle = 0
    for k in range(iterations): # there where 7 groups in topo_group
        run_nand_netlist(mem, toggle, 1-toggle, netlist_nand)
        toggle = 1-toggle
    
    sum_value = 0
    for k in range(num_bits):
        sum_value += (1 << k) * mem[toggle, address_sum + k]
    sum_value += (1 << num_bits) * mem[toggle, address_carry]
    return sum_value


# In[614]:


bin((1<<63)-1 + (1<<63))


# In[615]:


@numba.jit
def nb_perform_add_parallel(sum_arr, a, b, num_bits, iterations, mem, address_a, address_b, address_sum, address_carry, netlist_nand):
    #mem*=0
    parallel = len(sum_arr)
    #one = (1 << parallel)-1
    #avoid overflow with more complicated formula:
    one = ((1 << (parallel-1))-1)+(1 << (parallel-1))
    for k in range(num_bits):
        mem[:,address_a+k] = one if (a & (1 << k)) != 0 else 0
        mem[:,address_b+k] = one if (b & (1 << k)) != 0 else 0
    
    toggle = 0
    for k in range(iterations): # there where 7 groups in topo_group
        run_nand_netlist(mem, toggle, 1-toggle, netlist_nand)
        toggle = 1-toggle
    
    for i in range(parallel):
        sum_arr[i] = 0
    for k in range(num_bits):
        for i in range(parallel):
            sum_arr[i] += (1 << k) * (0b1 & (mem[toggle, address_sum + k] >> i))
    for i in range(parallel):
        sum_arr[i] += (1 << num_bits) * (0b1 & (mem[toggle, address_carry] >> i))


# In[594]:


args = address_map[Symbol("a0")], address_map[Symbol("b0")], address_map[Symbol("sum0")], address_map[Symbol("carry_out0")], netlist_arrs[BooleanNand]


# In[593]:


mem[:,address_map[Symbol("a4")]]


# In[699]:


nb_perform_add(45, 12, 64, 43, mem, args[0], args[1], args[2], args[3], args[4])


# In[516]:


get_ipython().run_line_magic('timeit', 'operator.add(45,12)')


# In[701]:


sum_arr = np.zeros(shape=(64), dtype=np.uint32)


# In[702]:


get_ipython().run_line_magic('timeit', 'nb_perform_add_parallel(sum_arr, 45, 12, 64, 43, mem, args[0], args[1], args[2], args[3], args[4])')


# In[703]:


sum_arr


# In[511]:


for a in range(256):
    for b in range(256):
        #sum_ab = perform_add(a,b)
        sum_ab = nb_perform_add(
            a, b,
            8, mem,
            address_map[Symbol("a0")], address_map[Symbol("b0")], 
            address_map[Symbol("sum0")], address_map[Symbol("carry_out0")],
            netlist_arrs[BooleanNand]
        )
        assert(sum_ab == a+b)


# In[484]:


perform_add(45,12)


# In[623]:


#address_map


# In[ ]:





# In[1020]:


shader_code = """
#version 440

#define GROUPSIZE_X ##group_size_x##
#define GROUPSIZE_Y ##group_size_y##
#define GROUPSIZE_Z ##group_size_z##
#define GROUPSIZE (GROUPSIZE_X * GROUPSIZE_Y * GROUPSIZE_Z)
layout(local_size_x=GROUPSIZE_X, local_size_y=GROUPSIZE_Y, local_size_z=GROUPSIZE_Z) in;

layout (std430, binding=0) buffer buf_0 { uint mem[]; };
//layout (std430, binding=1) buffer buf_1 { uint mem_out[]; };
layout (std430, binding=2) buffer buf_2 { uint netlist[]; };

uniform uint offset_mem_in = 0;
uniform uint offset_mem_out = 0;
uniform uint offset_netlist = 0;
uniform uint stride_netlist = 0;
uniform uint num_items = 0;

void main()
{
    uint workgroup_idx = 
        gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y +
        gl_WorkGroupID.y * gl_NumWorkGroups.x +
        gl_WorkGroupID.x;
    uint global_idx = gl_LocalInvocationIndex + workgroup_idx * GROUPSIZE;
    if (global_idx >= num_items) return;
    uint idx_netlist = offset_netlist + global_idx*stride_netlist;
    uint idx_out = netlist[idx_netlist+0];
    uint idx_a = netlist[idx_netlist+1];
    uint idx_b = netlist[idx_netlist+2];
    uint a = mem[offset_mem_in + idx_a];
    uint b = mem[offset_mem_in + idx_b];
    mem[offset_mem_out + idx_out] = ##CODE##;
}
"""


# In[1021]:


shader_code_nand = (
    shader_code
    .replace("##CODE##", "~(a & b)")
    .replace("##group_size_x##", "128")
    .replace("##group_size_y##", "1")
    .replace("##group_size_z##", "1")
)


# In[ ]:





# In[1022]:


print(shader_code_nand)


# In[975]:


ctx = moderngl.create_context(standalone=True, require=440)


# In[1023]:


shader_nand = ctx.compute_shader(shader_code_nand)


# In[1024]:


buf_mem = ctx.buffer(data=mem.flatten())


# In[1025]:


buf_netlist_nand = ctx.buffer(data=netlist_arrs[BooleanNand].flatten())


# In[1026]:


buf_mem.bind_to_storage_buffer(0)


# In[1006]:


buf_mem.bind_to_storage_buffer(1)


# In[1027]:


buf_netlist_nand.bind_to_storage_buffer(2)


# In[1028]:


netlist_arrs[BooleanNand].shape, netlist_arrs[BooleanNand].dtype


# In[1029]:


mem.shape, mem.dtype


# In[1030]:


address_map[Symbol("b0")]


# In[1095]:


mem *= 0


# In[1096]:


nb_write_number_bits(324234,64,32,mem,address_map[Symbol("a0")])
nb_write_number_bits(2321,64,32,mem,address_map[Symbol("b0")])


# In[1097]:


buf_mem.write(mem.tobytes())


# In[1098]:


shader_nand["num_items"] = netlist_arrs[BooleanNand].shape[0]
shader_nand["offset_mem_in"] = 0
shader_nand["offset_mem_out"] = mem.shape[1]
shader_nand["offset_netlist"] = 0
shader_nand["stride_netlist"] = netlist_arrs[BooleanNand].shape[1]


# In[1099]:


mem.dtype, mem.shape


# In[1100]:


int(math.ceil(shader_nand["num_items"].value / 1024))


# In[1101]:


import time


# In[1102]:


toggle = 0
count = 0


# In[1103]:



for k in range(200):
    shader_nand["offset_mem_in"] = mem.shape[1] * toggle
    shader_nand["offset_mem_out"] = mem.shape[1] * (1-toggle)
    shader_nand.run(int(math.ceil(shader_nand["num_items"].value / 128)))
    #time.sleep(0.1)
    toggle = 1-toggle
    count += 1


# In[1104]:


count


# In[1105]:


download_mem = np.frombuffer(buf_mem.read(mem.shape[1]*2*4), dtype=mem.dtype)


# In[1106]:


download_mem.shape


# In[1107]:


address_map[Symbol("sum0")]


# In[1108]:


(download_mem[(mem.shape[1] * (toggle) + address_map[Symbol("sum0")]):][:63]), (download_mem[(mem.shape[1] * (1-toggle) + address_map[Symbol("sum0")]):][:63])


# In[889]:


bin(324234+2321)


# In[719]:


bin(57)


# In[795]:


download_netlist_nand = np.frombuffer(buf_netlist_nand.read(), dtype=netlist_arrs[BooleanNand].dtype)


# In[675]:


buf_netlist_nand.read()


# In[ ]:




