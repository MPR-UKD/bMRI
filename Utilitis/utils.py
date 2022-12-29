

def get_function_parameter(f):
    return f.__code__.co_varnames[:f.__code__.co_argcount][1:]