import numpy as np
from src.berger import maxtree, canonize_tree_no_order
from src.display import display_graph

image4 = np.array([
    [0, 1, 1],
    [0, 2, 1],
    [3, 2, 0]
], dtype=np.uint8)

print(image4)

# Compute the maxtree
parentt = maxtree(image4)
display_graph(parentt, image4, "images/test_maxtree")

# Canonize the maxtree
canonize_tree_no_order(parentt, image4)
display_graph(parentt, image4, "images/test_cannonic")

print(parentt)
