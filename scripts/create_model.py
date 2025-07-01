# Pass the model name as an argument in during when using e.g
#
# python create_model.py MODEL_NAME
#
# or
#
# python scripts/create_model.py MODEL_NAME

import os
import sys

args = sys.argv
path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

Model_dir = os.path.join(path,"Model","Dev")

name = args[1]

init_template = f"""from .Trader import {name}
trader = {name}()
"""

trader_template = f"""from Model.standard_template import Trader,export

class {name}(Trader):
    pass
"""

ParentDir = os.path.join(Model_dir,name)
os.makedirs(ParentDir, exist_ok=True)

with open(os.path.join(ParentDir,"__init__.py"),"x") as fio:
    fio.write(init_template)
with open(os.path.join(ParentDir,"Trader.py"),"x") as fio:
    fio.write(trader_template)