from collections import defaultdict

from grqe.query import *


def optimize(root: Node) -> Node:
    root = unfuse_leaves(root)
    root = rewrite(root)
    return share(root)


def share(root: Node) -> Node:
    cached: list[Node] = []

    def rec(node: Node) -> Node:
        for cache in cached:
            if cache == node:
                return cache

        children = (rec(c) for c in node.children())
        if isinstance(node, Lookup):
            res = node
        else:
            res = node.construct(children)
        cached.append(res)
        return res

    return rec(root)


def rewrite(node):
    if node.arity == 0:
        return node

    children = (rewrite(c) for c in node.children())
    node = node.construct(children)

    while True:
        rewritten = apply_one_rewrite(node)
        if rewritten is node:
            return canonical(node)
        node = canonical(rewritten)


DISTRIBUTES: dict[type, set[type]] = {
    Sequence: {Conjunction, Disjunction, Alternative},
    Conjunction: {Disjunction},
}


def apply_one_rewrite(node: Node) -> Node:
    distributing_types = DISTRIBUTES.get(type(node))
    if not distributing_types:
        return node

    for i, child in enumerate(node.children()):
        if type(child) in distributing_types:
            return distribute(node, i)
    return node


def distribute(node: Node, index: int) -> Node:
    children = list(node.children())
    promoted = children[index]

    expanded_children: list[list[Node]] = []
    for distributed_child in promoted.children():
        cur = list(children)
        cur[index] = distributed_child
        expanded_children.append(cur)

    return promoted.construct(node.construct(c) for c in expanded_children)


# Canonical form
# Remove neutral/absorbing elements (only Epsilon within Sequences)
# Flatten nested (relies on Assoc)
# Deduplicate operands (relies on Idemp)
# Sort children to canonical form (relies on Comm)

def canonical(root: Node) -> Node:
    return order_children(flatten_associative(remove_neutral_elements(root)))


def order_children(root: Node) -> Node:
    if isinstance(root, Lookup):
        return Lookup(root.width, tuple(sorted(root.atoms)))

    if root.arity == 0:
        return root

    children = (canonical(c) for c in root.children())
    if root.is_commutative & root.is_idempotent:
        # Sort AND deduplicate children.
        children = sorted(set(children))
    return root.construct(children)


# Canonical right now does not have different paths for idempotent or commutative ops.
assert all(op.is_commutative == op.is_idempotent for op in Node.OPERATOR_TYPES), \
    f'Implementation of {order_children.__name__} relies on all idempotent operators to also be commutative.'


def remove_neutral_elements(root: Node) -> Node:
    if root.arity == 0:
        return root

    if isinstance(root, Sequence):
        elements = tuple(remove_neutral_elements(c) for c in root.elements if not isinstance(c, Epsilon))
        match len(elements):
            case 0:
                return Epsilon()
            case 1:
                return elements[0]
            case _:
                return Sequence(elements)
    else:
        children = (remove_neutral_elements(c) for c in root.children())
        return root.construct(children)


def flatten_associative(root: Node) -> Node:
    if root.arity == 0:
        return root

    root_type = type(root)

    def _unpack(node: Node) -> Generator[Node]:
        for c in node.children():
            if isinstance(c, root_type):
                yield from _unpack(c)
            else:
                yield c

    if root_type.is_associative:
        children = list(_unpack(root))
    else:
        children = root.children()

    rec = (flatten_associative(c) for c in children)
    return root_type.construct(rec)


def unfuse_leaves(root: Node) -> Node:
    if isinstance(root, Lookup):
        leaves = defaultdict(set)

        for atom in root.atoms:
            leaves[atom.relative_position].add((atom.key, atom.value))

        min_offset = min(leaves.keys())
        max_offset = max(leaves.keys())
        sequence = []
        for offset in range(min_offset, max_offset + 1):
            entries_at_offset = leaves[offset]
            if not entries_at_offset:
                node = Arbitrary()
            else:
                atoms = (Atom(0, key, value) for key, value in entries_at_offset)
                node = Lookup(tuple(atoms))
            sequence.append(node)
        return Sequence(tuple(sequence))
    else:
        children = (unfuse_leaves(c) for c in root.children())
        return root.construct(children)
