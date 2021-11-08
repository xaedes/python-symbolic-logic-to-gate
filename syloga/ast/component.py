
import itertools
from enum import Enum

from dataclasses import dataclass

from syloga.ast.core import Expression
from syloga.ast.containers import Dict
from syloga.ast.function import FunctionArgument
from syloga.ast.function import FunctionDeclaration
from syloga.ast.function import FunctionDefinition

class ComponentSection(Enum):
    Input = 0
    Output = 1
    Local = 2
    Data = 3
    Rest = 4


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
        
