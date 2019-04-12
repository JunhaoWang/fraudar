# Runs the greedy algorithm.
# Command line arguments: first argument is path to data file. Second argument is path to save output. Third argument (*optional*) is path where the node suspiciousness values are stored.
import time

import numpy as np

start_time = time.time()
from greedy import readData, logWeightedAveDegree
import sys

M = readData(sys.argv[1])
# The example data is a 500 x 500 matrix with an injected dense block among the first 20 nodes
print(("finished reading data: shape = %d, %d @ %d" % (M.shape[0], M.shape[1], time.time() - start_time)))

lwRes = logWeightedAveDegree(M)

print(lwRes)
np.savetxt("%s.rows" % (sys.argv[2], ), np.array(list(lwRes[0][0])), fmt='%d')
np.savetxt("%s.cols" % (sys.argv[2], ), np.array(list(lwRes[0][1])), fmt='%d')
print(("score obtained is %f" % (lwRes[1],)))
print(("done @ %f" % (time.time() - start_time,)))