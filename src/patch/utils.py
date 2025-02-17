
def patch(cls, name=None):
    def apply(func):
        nonlocal name
        if name == None:
            name = func.__name__
        setattr(cls, name, func)
        return func
    return apply