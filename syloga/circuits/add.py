
import itertools

from syloga.ast.Tuple import Tuple
from syloga.ast.BreakOut import BreakOut

from syloga.ast.core import numbered_symbols

def half_adder(a, b):
    half_sum = (a ^ b)
    carry = (a & b)
    return Tuple(half_sum, carry)

def full_adder(a, b, c):
    sum_ab, carry_ab = half_adder(a, b)
    sum_abc, carry_abc = half_adder(sum_ab, c)
    
    return Tuple(sum_abc, carry_ab | carry_abc)

def add_cascade(a, b, carry_in):
    carry = carry_in
    result = []
    for a_bit, b_bit in zip(a,b):
        sum_bit, carry = full_adder(a_bit, b_bit, carry)
        #carry = BreakOut(carry)
        result.append(sum_bit)

    sum_bits = Tuple(*map(lambda x:BreakOut(x,"sum"),result))
    carry_out = BreakOut(carry,"carry_out")
    
    return Tuple(sum_bits, carry_out)

def add_cascade_symboled(n,a="a",b="b"):
    return add_cascade(
        list(numbered_symbols(a,0,n)), 
        list(numbered_symbols(b,0,n)), 
        Symbol("carry")
    )


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
        GP_dict.update(dict(zip(
            idcs[0], 
            itertools.starmap(
                lambda k0,k1: (
                    union_of_idcs(k0,k1),
                    to_op(k0,k1)
                ), 
                zip(*idcs)
            )
        )))

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


def brent_kung_cla_symboled(n,a="a",b="b"):
    return brent_kung_cla(
        list(numbered_symbols(a,0,n)), 
        list(numbered_symbols(b,0,n))
    )

