
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

