from rtree import RTree, load_sample_file

# load the 10 sample points + the query point for Task 2
data, queries = load_sample_file("datasets/Sample_Data/NN_BF_search.txt")
print("Data:", data)
print("Query:", queries)

# build the R-tree with B=4 and print every insertion step
tree = RTree(B=4, verbose=True)
tree.build(data)