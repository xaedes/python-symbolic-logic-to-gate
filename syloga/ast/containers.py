
from dataclasses import dataclass
from syloga.ast.core import Expression

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
