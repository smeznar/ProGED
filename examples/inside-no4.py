print(1)
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

# from common.aaa import say_hi
# import sys
# sys.path.append('.')
# sys.path.append('..')
 
import useful
# from common.aaa import say_hikk
# from PCFGproject import useful 
# from .. import useful 
print("after import in inside")
