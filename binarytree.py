"""
binarytree package

https://pypi.org/project/binarytree/
"""

#https://binarytree.readthedocs.io/en/main/specs.html#class-binarytree-node
from binarytree import Node

root = Node(1)
root.left = Node(3)
root.right = Node(2)
root.left.right = Node(5)
root.left.right.left = Node(8)
root.left.right.right = Node(6)

# print the list of nodes
list(root)

# using print to visualize the tree
print(root)

# Tree height at any nodes
root.height
root.left.height

# check to see if any node is balanced
root.is_balanced
root.left.right.is_balanced

# check to see if any node is binary search tree
root.is_bst

# check to see if any node is 
root.is_complete

# check to see if any node is 
root.is_max_heap

# check to see if any node is 
root.is_min_heap

# check to see if any node is 
root.is_perfect

# check to see if any node is 
root.is_strict

# count the number of leaf nodes
root.leaf_count

#  maximum leaf depth (the farthest leaf)
root.max_leaf_depth

# max node values
root.max_node_value

# minimum leaf depth (the closest leaf)
root.min_leaf_depth

# min node values
root.min_node_value

# count the number of nodes
root.size
