# debug_imports.py  (place in repo root)
import importlib.util as iu
import pkgutil, sys, pathlib, textwrap

print("\n‣ Current working directory:")
print(" ", pathlib.Path().resolve())

print("\n‣ First 10 entries on sys.path:")
for p in sys.path[:10]:
    print(" ", p)

print("\n‣ Does Python see 'regiem_45_model'?")
spec = iu.find_spec("regiem_45_model")
print(" ", "FOUND at →" if spec else "NOT FOUND", spec.origin if spec else "")

if spec:
    pkg = iu.module_from_spec(spec)
    spec.loader.exec_module(pkg)
    print("\n‣ Sub-modules inside regiem_45_model:")
    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    print(textwrap.fill(", ".join(names), width=80))
