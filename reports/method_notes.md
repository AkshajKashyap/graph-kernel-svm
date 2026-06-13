# Method Notes

## Graph-Stat Baseline

The graph-stat baseline maps each graph to a small vector of global structural
measurements: node count, edge count, average degree, density, triangle count, and
connected-component count. The kernel is the linear dot product between these vectors.
It is inexpensive and useful as a reference, but it discards most local structure.

## Weisfeiler-Lehman Subtree Kernel

The Weisfeiler-Lehman (WL) subtree kernel starts from discrete node labels. At each
iteration, a node receives a deterministic compressed label derived from its current
label and the sorted labels of its neighbors. Label frequencies from every iteration
form the graph feature vector. The kernel is the dot product of those feature counts.
The implementation evaluates depths zero through five.

## Shortest-Path Kernel

The shortest-path kernel runs breadth-first search from every node. Each reachable,
unordered node pair contributes a feature containing the sorted endpoint labels and
their shortest-path distance. The resulting feature counts preserve distance-based
structure while remaining invariant to node ordering.

## Precomputed SVM

The project uses `SVC(kernel="precomputed")` because each custom graph kernel produces
an explicit graph-by-graph similarity matrix. For an outer split, training receives
`K[train, train]` and prediction receives `K[test, train]`. This separates kernel
construction from the classifier and makes comparisons use the same SVM interface.

## Leakage-Safe C Tuning

The SVM regularization parameter `C` is selected independently inside every outer
training split. Stratified inner folds are built only from the outer training indices.
Validation matrices are sliced as `K[inner_validation, inner_train]`. The selected
value is then refit on the full outer training block before evaluating the untouched
outer test block.

## Repeated Stratified Splits

Repeated stratified train/test splits preserve class proportions while measuring
variation caused by sampling. Reports include the mean and standard deviation of
accuracy and macro F1. These estimates are more informative than one split, although
they are not a substitute for a larger benchmark protocol.

## Limitations

- The kernels are implemented for clarity rather than large-scale efficiency.
- Node labels are discrete; continuous attributes and edge labels are not modeled.
- Hyperparameter tuning currently covers only SVM `C` and WL depth comparisons.
- TU datasets are small and results can vary noticeably across random splits.
- The project does not claim state-of-the-art performance.
