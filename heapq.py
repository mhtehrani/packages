"""
heapq package
"""

"""
Heap:

It's a balanced binary tree

* There are two types, Max and Min.
* In the Min Heap, the root is the smallest element and as go down the trees, the values get bigger and bigger.
* It can best constructed with a 1-D array, the root has index 0
* The childs of node i are 2*i+1 and 2*i+2
* The parents of node i is (i-1)//2


"""

import heapq #only min heap
heap = []

# Push the value item onto the heap, maintaining the heap invariant.
heapq.heappush(heap, item)

# Pop and return the smallest item from the heap, maintaining the heap invariant. If the heap is empty, IndexError is raised. To access the smallest item without popping it, use heap[0].
heapq.heappop(heap)

# Transform list x into a heap, in-place, in linear time.
heapq.heapify(heap)

# Return a list with the n largest elements from the dataset defined by iterable. key, if provided, specifies a function of one argument that is used to extract a comparison key from each element in iterable (for example, key=str.lower). Equivalent to: sorted(iterable, key=key, reverse=True)[:n].
heapq.nlargest(n, iterable, key=None)

# Return a list with the n smallest elements from the dataset defined by iterable. key, if provided, specifies a function of one argument that is used to extract a comparison key from each element in iterable (for example, key=str.lower). Equivalent to: sorted(iterable, key=key)[:n].
heapq.nsmallest(n, iterable, key=None)