
import sys
import inspect
import importlib

print(f"Python: {sys.executable}")

try:
    import clouds_decoded.modules.refl2prop.model as m
    print(f"File: {m.__file__}")
    
    from clouds_decoded.modules.refl2prop.model import InversionNet
    print("Source of InversionNet.__init__:")
    print(inspect.getsource(InversionNet.__init__))
except Exception as e:
    print(e)
