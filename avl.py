
if __name__ == "__main__":
    from utils import *

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1
        self.subtree_sum_output_length = key.output_length
        self.subtree_sum_bandwidth = key.get_bandwidth()
        self.epoch = 0

class AVLTree:
    def __init__(self):
        self.root = None

    def _height(self, node):
        return node.height if node else 0

    def _output_sum(self, node):
        return node.subtree_sum_output_length if node else 0

    def _bandwidth_sum(self, node):
        return node.subtree_sum_bandwidth if node else 0

    def _update(self, node):
        node.height = 1 + max(self._height(node.left), self._height(node.right))
        node.subtree_sum_output_length = node.key.output_length + self._output_sum(node.left) + self._output_sum(node.right)
        node.subtree_sum_bandwidth = node.key.get_bandwidth() + \
            self._bandwidth_sum(node.left) + self._bandwidth_sum(node.right)

    def _balance_factor(self, node):
        return self._height(node.left) - self._height(node.right)

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        self._update(y)
        self._update(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        self._update(x)
        self._update(y)
        return y

    def _balance(self, node):
        self._update(node)
        bf = self._balance_factor(node)

        if bf > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if bf < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if not node:
            return AVLNode(key)
        if key.output_length < node.key.output_length:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        return self._balance(node)

    def get_output_length_sum_total_less_than(self, x):
        return self._get_output_sum(self.root, x)
    
    def get_bandwidth_sum_total_less_than(self, x):
        return self._get_bandwidth_sum(self.root, x)

    def get_all_less_than(self, x):
        return self._get_less_than(self.root, x)
    
    def _get_less_than(self, node, x):
        if not node:
            return []
        if x <= node.key.output_length:
            return self._get_less_than(node.left, x)
        else:
            return self._get_less_than(node.left, x) + [node.key] + self._get_less_than(node.right, x)

    def _get_output_sum(self, node, x):
        if not node:
            return 0
        if x <= node.key.output_length:
            return self._get_output_sum(node.left, x)
        else:
            return node.key.output_length + self._output_sum(node.left) + self._get_output_sum(node.right, x)
    
    def _get_bandwidth_sum(self, node, x):
        if not node:
            return 0
        if x <= node.key.get_bandwidth():
            return self._get_bandwidth_sum(node.left, x)
        else:
            return node.key.get_bandwidth() + \
                   self._bandwidth_sum(node.left) + \
                   self._get_bandwidth_sum(node.right, x)

