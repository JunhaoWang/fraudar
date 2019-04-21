# Runs the greedy algorithm.
# Command line arguments: first argument is path to data file. Second argument is path to save output. Third argument (*optional*) is path where the node suspiciousness values are stored.
import time

import numpy as np

start_time = time.time()
from greedy import readData, logWeightedAveDegree
import sys
from sklearn.metrics import f1_score


M = readData(sys.argv[1])

# Now we inject anomalies into the adjacency matrix

print(("finished reading data: shape = %d, %d @ %d" % (M.shape[0], M.shape[1], time.time() - start_time)))

lwRes = logWeightedAveDegree(M)

# Compute y_pred
y_true = np.load(sys.argv[2])
anons = np.array(list(lwRes[0][0]))
y_pred = np.zeros_like(y_true)
y_pred[anons] = 1

# Save y_pred
np.save('pred_anomalies_fraudar.npy', y_pred)

# f1 = f1_score(y_true, y_pred)
# print("The final F1 score is: %.4f" % f1)

# print(lwRes)
# np.savetxt("%s.rows" % (sys.argv[3], ), np.array(list(lwRes[0][0])), fmt='%d')
# np.savetxt("%s.cols" % (sys.argv[3], ), np.array(list(lwRes[0][1])), fmt='%d')
# print(("score obtained is %f" % (lwRes[1],)))
# print(("done @ %f" % (time.time() - start_time,)))