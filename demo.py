import dgl
import numpy as np
import torch as th
from dgl.nn import GATv2Conv

# Case 1: Homogeneous graph
g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
g = dgl.add_self_loop(g)
feat = th.ones(10)
gatv2conv = GATv2Conv(10, 5, num_heads=1)
res = gatv2conv(g, feat)
print(res.shape)