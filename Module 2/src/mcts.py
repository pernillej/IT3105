"""
TODOs
1. Tree Search - Traversing the tree from the root to a leaf node by using the tree policy.
2. Node Expansion - Generating some or all child states of a parent state, and then connecting the tree node housing
the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
3. Leaf Evaluation - Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
policy from the leaf node’s state to a final state.
4. Backpropagation - Passing the evaluation of a final state back up the tree, updating relevant data
(see course lecture notes) at all nodes and edges on the path from the final state to the tree root.

"""


class MCTS:

    def __init__(self):
        return
