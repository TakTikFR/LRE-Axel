"""
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
print("other: ")
for i in range(len(parentt)):
    for j in range(len(parentt[0])):
        print(image4[i][j], end=" ")
    print('\n')
print('\n')
"""

from src.berger import maxtree, canonize_tree_no_order
from src.display import display_graph
import time
import cv2
import numpy as np
import os

def load_gray_image_as_array(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def main():
    for i in range(1, 8):
        #path = f"/home/axel/Documents/LRE/LRE-Axel/cuda/images/1080p/1080p_{i}.png"

        #if not os.path.exists(path):
        #    print(f"{path} n'existe pas, on saute.")
        #    continue

        #f = load_gray_image_as_array(path)

        #start = time.time()

        f = np.array([
            255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
], dtype=np.uint8).reshape(16, 16)

        parent = maxtree(f)
        parent = canonize_tree_no_order(parent, f)
        display_graph(parent, f, "verif");

        #end = time.time()
        #duration = end - start

        #print(f"{path} : Temps d'exécution : {duration:.6f} secondes")

if __name__ == "__main__":
    main()