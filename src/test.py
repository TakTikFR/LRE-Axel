from Berger import *

image4 = np.array([[0, 1, 1], [0, 2, 1], [3, 2, 0]], dtype=np.uint8)
print(image4)
parentt = maxtree(image4)
display_graph(parentt, image4, "test2")
canonize_tree_noR(parentt, image4)
display_graph(parentt, image4, "test3")
print(parentt)
