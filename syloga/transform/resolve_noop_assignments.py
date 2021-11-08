
from syloga.utils.compose import compose
from syloga.core.decide_predicate_usage import decide_predicate_usage
from syloga.transform.replace import replace
from syloga.transform.topo_sort import topo_sort

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
    assigned_symbols = list(map(
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
            partial(filter,lambda x:is_not_excluded(x) and must_not_keep(x) and x not in assigned_symbols),
            #partial(filter,must_not_keep),
            
        ),groups
    ))
    
    # build replacement pairs
    replacements = dict([
        (replace, by)
        for replace_list, by in zip(replaced_symbols, assigned_symbols)
        if by is not None
        for replace in replace_list
    ])
    
    #print("sorted_lhs", sorted_lhs)
    #print("noop_assignments", noop_assignments)
    #print("pairs", pairs)
    #print("groups", groups)
    #print("assigned_symbols", assigned_symbols)
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

