
import itertools
from dataclasses import dataclass

from syloga.ast.core import Expression
from syloga.ast.core import Symbol
from syloga.ast.core import is_lvalue
from syloga.ast.containers import Tuple
from syloga.ast.containers import Dict
from syloga.ast.traversal import zip_expression
from syloga.transform.basic import map_expression_args
from syloga.utils.predicates import decide_predicate_usage

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
        # so far i don't know how to leverage inout (only in xor out), but it seems useful to keep for future use
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
        return Tuple(*(
            ((num_bound_pos_args+k, arg) if with_idx else arg)
            for k,arg in enumerate(self.function_declaration.arguments[num_bound_pos_args:])
            if arg.name not in self.kwarguments
        ))
    
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
    

    
    def list_dependees(self):
        return list(filter(lambda x:x.is_out,self.arguments))
    
    def list_dependencies(self):
        return list(filter(lambda x:x.is_in,self.arguments))    
    
    def inline(self, function_definition):
        assert(function_definition.function_declaration == self.function_declaration)
        return function_definition.inline(self)
    
    def assign_to(self, lhs):
        """make new FunctionCall with lhs assigned to unbound output arguments."""
        kwarguments = dict(*self.kwarguments.items())
        unbound_out = self.unbound_output_arguments()
        if type(lhs) != Tuple:
            lhs = Tuple(lhs)
        # TODO: it would be nice to support assign_to Dict 
        kwarguments.update(
            zip_expression(
                tuple(map(
                    lambda x:x.name,
                    unbound_out
                )),
                lhs,
                key=is_lvalue,
                flat=True,
                as_list=False
            )        
        )
        return FunctionCall(
            self.function_declaration,
            Tuple(*self.arguments),
            kwarguments
        )
        
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
    
    def list_dependees(self):
        return list(filter(lambda x:x.is_out,self.arguments))
    
    def list_dependencies(self):
        return list(filter(lambda x:x.is_in,self.arguments))
    
    def call(self, *args, **kwargs):
        return FunctionCall(self, args, kwargs)


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
                    else map_expression(r,x)
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

