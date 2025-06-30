import sys, os, importlib, tomllib
import os, tomllib

config_path = os.path.abspath("config.toml")  # always relative to CWD
with open(config_path, "rb") as f:
    config = tomllib.load(f)
activeModel = config['Development-Settings']['ActiveModel']

# Add model root (dev or build) to sys.path
model_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, model_dir)

# Import the active model normally
ActiveModel = importlib.import_module(activeModel)
