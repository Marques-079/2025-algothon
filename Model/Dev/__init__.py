import sys, os, importlib, tomllib
import os, tomllib

config_path = os.path.abspath("config.toml")  # always relative to CWD
with open(config_path, "rb") as f:
    config = tomllib.load(f)
activeModel = config['Development-Settings']['ActiveModel']

# Add model root (dev or build) to sys.path
model_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, model_dir)

def list_package_dirs(base_path):
    """List all subdirectories under `base_path` that contain an __init__.py."""
    return [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name)) and
           os.path.isfile(os.path.join(base_path, name, '__init__.py'))
    ]

valid_models = list_package_dirs(model_dir)
if activeModel in valid_models:
# Import the active model normally
    ActiveModel = importlib.import_module(activeModel)
else:
    print("╭───────────────────────────────────────────────╮")
    print( "│\x1b[38;2;255;102;102mYou did not select a model, select one now! ⚠️  \x1b[0m│" )
    print("╰───────────────────────────────────────────────╯")
    print()
    for i,v in enumerate(valid_models):
        print(f"\033[3m - Model {i}\033[0m: {v}")
    print()
    no = int(input("\033[38;2;96;214;149mModel to run (model number)\033[0m:"))
    ActiveModel = importlib.import_module(valid_models[no])

