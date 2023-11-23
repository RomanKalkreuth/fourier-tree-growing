import sys

sys.path.insert(0, '../gp/gpsimple')
sys.path.insert(0, '../distance')

import random
import numpy as np
import plotly.express as px
import distance as distance
from gp_tree import GPNode

random.seed()
np.random.seed()

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 6
SUBTREE_DEPTH = 4
MUTATION_RATE = 0.1
NUM_SAMPLES = 10000
STEP = 0.01
N = int(1.0 / STEP) + 1

x = np.empty(N)
y = np.empty(N)

rate = 0
index = 0
while rate <= 1.0:
    distances = np.empty(NUM_SAMPLES)
    for i in range(NUM_SAMPLES):
        t1 = GPNode()
        t1.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)

        t2 = t1.clone()
        t2.mutate(mutation_rate=rate, subtree_depth=SUBTREE_DEPTH)

        # ld = distance.levenshtein_distance(t1, t2)
        ted = distance.tree_edit_distance(t1, t2)
        distances[i] = ted

    y_val = np.mean(distances)
    x_val = round(rate, 2)

    x[index] = x_val
    y[index] = y_val

    print(str(x_val) + " " + str(y_val))

    index += 1
    rate += STEP

# Levenshtein distance
plot = px.scatter(x=x, y=y, labels={
    "x": "Mutation rate",
    "y": "Tree edit distance",
}, title="Mutation strength/distance correlation")

plot.update_layout(yaxis=dict(titlefont=dict(size=25), tickfont=dict(size=20)),
                   xaxis=dict(titlefont=dict(size=25), tickfont=dict(size=20)),
                   title=dict(font=dict(size=25)),
                   yaxis_range=[0, 50])

plot.show()
