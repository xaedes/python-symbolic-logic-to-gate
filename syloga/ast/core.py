
from dataclasses import dataclass

@dataclass
class Expression:
    func: type
    args: tuple
    def __init__(self, func, args):
        self.func = func
        self.args = tuple(args)
    
    def copy(self):
        if self.func is not Expression:
            return self.func(*self.args)
        else:
            return Expression(self.func, self.args)

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
    
    def __or__( self, other):
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

def symbols(string):
    return Tuple(*map(Symbol,string.split(" ")))

def numbered_symbols(string="x", start_idx=0, count=None, interval=1, infix=""):
    k = 0
    while count is None or k < count:
        idx = start_idx + interval * k
        yield Symbol(string + infix + str(idx))
        k += 1

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
class Indexable(Expression):
    def __init__(self, name:str):
        self.name = name
        super().__init__(Indexable, (self.name, ))
        
    def __hash__(self):
        return super().__hash__()
    
    def __repr__(self):
        return self.name
    
    def __getitem__(self, index:int):
        return Indexed(self, index)

def is_lvalue(x):
    return isinstance(x, (Symbol, Indexed, Indexable))

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
class BooleanExpression(Expression):
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

