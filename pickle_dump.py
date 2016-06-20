import pickle
import sys
from pprint import pprint

with open(sys.argv[1], 'rb') as fp:
    b = pickle.load(fp)

if sys.argv[2] == 'name':
    print b
else:
    for p in b:
        pprint(p.get_value())
        print " ------ "
