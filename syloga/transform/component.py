
import typing
import dataclasses
from dataclasses import dataclass

from syloga.ast.component import ComponentSection
from syloga.ast.component import ExpressionComponent
from syloga.ast.containers import Dict
from syloga.ast.containers import Tuple
from syloga.ast.core import Assignment

from syloga.utils.misc import split_name_id

from syloga.transform.replace import replace


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
    # print(assignments)
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
