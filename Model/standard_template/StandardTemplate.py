import inspect
import types

def export(func):
    """ Add this as a decorator to a function or method which will be exported as getMyPosition"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper._is_export = True  # Add custom attribute
    return wrapper

def export_trading_funcs():
    """ Call to export a tradding function with the @export decorator as getMyPosition for evaluation
    Note, the first function with the decorator will be exported."""
    frame = inspect.currentframe()
    caller_globals = frame.f_back.f_globals
    for name,f in caller_globals.items():
        if isinstance(f, (types.FunctionType, types.MethodType)):
            if getattr(f,"_is_export",False):
                print(f"Exporting: {name}")
                exec(f"getMyPosition = {name}",caller_globals)
                return

# Function to scan a given namespace dict (locals() or globals())
# and find all functions with the _is_marked attribute
def find_exports(scope):
    marked_methods = []
    for name in dir(scope):
        attr = getattr(scope, name)
        if isinstance(attr, (types.FunctionType, types.MethodType)):
            if getattr(attr, '_is_export', False):
                marked_methods.append(name)
    return marked_methods


class Trader():
    """ Base class for building more complex traders that would hold tate
    "Using global too many variables is a bit stupid at that point guys " """
    def __init__(self):
        pass

    def export_trader(self):
        """ Exports trader method with @export decorator as getMyPosition for evaluation """
        frame = inspect.currentframe()
        cname = self.__class__
        caller_globals = frame.f_back.f_globals
        f = find_exports(self)
        for k,item in caller_globals.items():
            if isinstance(item,cname):
                print(f"Exporting: {k}.{f[0]}")
                exec(f"getMyPosition = {k}.{f[0]}",caller_globals)
                return
    
    def get_exported(self):
        """Return exported trader method with @export as a function"""
        f = find_exports(self)
        return getattr(self,f[0])