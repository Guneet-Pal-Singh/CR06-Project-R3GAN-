import re
p = 'torch_utils/custom_ops.py'
src = open(p).read()

if '_loaded_module' in src:
    raise SystemExit("already patched")

n_load = len(re.findall(r'torch\.utils\.cpp_extension\.load\(', src))
n_imp  = src.count('module = importlib.import_module(module_name)')
print(f'found {n_load} load() calls and {n_imp} import_module lines')
assert n_load >= 1 and n_imp == 1, "unexpected file layout - paste torch_utils/custom_ops.py here"

src = re.sub(r'(\n\s*)torch\.utils\.cpp_extension\.load\(', r'\1_loaded_module = torch.utils.cpp_extension.load(', src)
src = src.replace('module = importlib.import_module(module_name)', 'module = _loaded_module')
open(p, 'w').write(src)
print('patched')
