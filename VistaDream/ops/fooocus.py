import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/Fooocus'
sys.path.insert(0,reference)

from fooocus_command import Fooocus
