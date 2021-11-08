
import itertools

from syloga.transform.get_assignments import get_assignments
from syloga.ast.traversal.iter_dependees import iter_dependees
from syloga.ast.traversal.iter_dependencies import iter_dependencies
from syloga.utils.functional import compose
from syloga.utils.functional import iter_unique

from syloga.ast.core import Indexed
from syloga.ast.containers import Tuple

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

def list_generic_topo_group(iterable, get_dependees, get_dependencies):
    return list(iter_generic_topo_group(iterable, get_dependees, get_dependencies))

def topo_depth(expression):
    return len(topo_group(expression))

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

def topo_sort(assignments):
    groups = topo_group(assignments, sort_in_group=True)
    return Tuple(*itertools.chain(*groups))

