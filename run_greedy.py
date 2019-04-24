# Runs the greedy algorithm.
# Command line arguments: first argument is path to data file. Second argument is path to save output. Third argument (*optional*) is path where the node suspiciousness values are stored.
import time

import numpy as np

start_time = time.time()
from greedy import readData, logWeightedAveDegree
from scipy import sparse
import sys
from sklearn.metrics import f1_score
import networkx as nx

def generate_dense_block_ajacency(node_size, density):
    edge_size = int(node_size * (node_size-1) / 2 * density)
    return nx.adjacency_matrix(
        nx.generators.dense_gnm_random_graph(node_size,edge_size), weight=None
    ).A


# M = readData(sys.argv[1])
M = np.load(sys.argv[1])

# Now we inject anomalies into the adjacency matrix
blocks = [
        {'idx_range': (0, 1000), 'ajacency_density': float(sys.argv[2])},
        {'idx_range': (2000, 3000), 'ajacency_density': float(sys.argv[2])},
        {'idx_range': (6000, 7000), 'ajacency_density': float(sys.argv[2])},
    ]

for ind, b in enumerate(blocks):
    s_idx = b['idx_range'][0]
    e_idx = b['idx_range'][1]
    M[s_idx:e_idx, s_idx:e_idx] = generate_dense_block_ajacency(e_idx - s_idx, b['ajacency_density'])
M = sparse.coo_matrix(M)


# Setting labels to be 0 or 1 based on whether or not they correspond to an anomaly
input_size = M.shape[0]
labels = np.zeros((input_size, 1))
for block in blocks:
    idx_s, idx_e = block['idx_range']
    labels[idx_s:idx_e] = 1
labels = labels.astype(np.uint8)

print(("finished reading data: shape = %d, %d @ %d" % (M.shape[0], M.shape[1], time.time() - start_time)))

lwRes = logWeightedAveDegree(M)

# Compute y_pred
y_true = labels
anons = np.array(list(lwRes[0][0]))
y_pred = np.zeros_like(y_true)
y_pred[anons] = 1

f1 = f1_score(y_true, y_pred)
print("The final F1 score for a density of %.3f is: %.4f" % (float(sys.argv[2]), f1))

# print(lwRes)
# np.savetxt("%s.rows" % (sys.argv[3], ), np.array(list(lwRes[0][0])), fmt='%d')
# np.savetxt("%s.cols" % (sys.argv[3], ), np.array(list(lwRes[0][1])), fmt='%d')
# print(("score obtained is %f" % (lwRes[1],)))
# print(("done @ %f" % (time.time() - start_time,)))