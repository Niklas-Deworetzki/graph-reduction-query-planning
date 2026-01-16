from grqe.query import *


# Structural normalization
# Flatten nested (relies on Assoc)
# Deduplicate operands (relies on Idemp)
# Remove neutral/absorbing elements (requires Epsilon to be present)
# Sort children to canonical form (relies on Comm)

def canonical(root: Node) -> Node:
    if isinstance(root, Lookup):
        return Lookup(tuple(sorted(root.atoms)))

    if root.arity == 0:
        return root

    children = [canonical(c) for c in root.children()]
    if root.is_commutative & root.is_idempotent:
        # Sort AND deduplicate children.
        children = sorted(set(children))
    return root.construct(tuple(children))


# Canonical right now does not have different paths for idempotent or commutative ops.
assert all(op.is_commutative == op.is_idempotent for op in Node.OPERATOR_TYPES), \
    f'Implementation of {canonical.__name__} relies on all idempotent operators to also be commutative.'


def remove_neutral_elements(root: Node) -> Node:
    if root.arity == 0:
        return root

    if isinstance(root, Sequence):
        elements = [remove_neutral_elements(c) for c in root.elements if not isinstance(c, Epsilon)]
        if not elements:
            return Epsilon()
        return Sequence(elements)
    else:
        children = [remove_neutral_elements(c) for c in root.children()]
        return root.construct(children)


def flatten(root: Node) -> Node:
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
        if len(children) == 1:
            return flatten(children[0])
    else:
        children = root.children()

    rec = (flatten(c) for c in children)
    return root_type.construct(rec)
