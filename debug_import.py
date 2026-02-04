
import sys
import inspect
import torch

try:
    from clouds_decoded.modules.refl2prop.model import InversionNet, NormalizationWrapper
    
    print(f"Imported InversionNet from: {inspect.getfile(InversionNet)}")
    
    model = InversionNet()
    print("InversionNet structure:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module)}")
        
    wrapper = NormalizationWrapper(model, {'min':[], 'max':[]}, {'min':[], 'max':[]})
    print("\nWrapper State Dict Keys (First 5):")
    print(list(wrapper.state_dict().keys())[:5])

except Exception as e:
    print(f"Error: {e}")
