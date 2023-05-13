from dataclasses import dataclass

# from turtle import backward
from typing import Any, Iterable, List, Protocol, Tuple

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    """val = list(vals)
    minus = val[:]
    val[arg] = val[arg] + epsilon
    minus[arg] = minus[arg] - epsilon
    return (f(*val) - f(*minus)) / (2 * epsilon)"""
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    """sort = []

    def bfs(n: Variable) -> None:
        if n is None:
            return
        # queue= Queue()
        queue = []
        visit = set()
        # queue.app(n)
        queue.append(n)
        visit.add(n.unique_id)
        # while not queue.empty():
        while queue:
            # current = queue.get()
            current = queue.pop(0)
            sort.append(current)
            for p in current.parents:
                if p.unique_id not in visit:
                    visit.add(p.unique_id)
                    # queue.put(p)
                    queue.append(p)

    bfs(variable)
    return sort"""
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    """# Call topological sort to get an ordered queue
    sort = topological_sort(variable)
    # Create a dictionary of Scalars and current derivatives
    graph = {}
    for index, v in enumerate(sort):
        if index == 0:
            graph[v.unique_id] = deriv
        else:
            graph[v.unique_id] = 0.0

    # graph[sort[0].unique_id] = deriv
    # For each node in backward order, pull a completed Scalar and derivative from the queue
    for node in sort:
        # if the Scalar is a leaf, add its final derivative (accumulate_derivative) and loop to (1)
        if node.is_leaf():
            node.accumulate_derivative(graph[node.unique_id])
        else:
            # call .backprop_step on the last function with dout
            chain = node.chain_rule(graph[node.unique_id])
            # loop through all the Scalars+derivative produced by the chain rule
            for (vara, deri) in chain:
                graph[vara.unique_id] += deri
            # accumulate derivatives for the Scpytest -m task0_1alar in a dictionary"""
    queue = topological_sort(variable)
    deri = {}
    deri[variable.unique_id] = deriv
    for var in queue:
        deriv = deri[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                # get the value(derivative) of v.unique_id, if not exists, return 0
                deri.setdefault(v.unique_id, 0.0)
                deri[v.unique_id] = deri[v.unique_id] + d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
