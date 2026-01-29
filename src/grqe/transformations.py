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
# Deduplicate operands (relies on Idemp (and technically commutative and associative)
# Sort children to canonical form (relies on Comm)

def canonical(root: Node) -> Node:
    return order_children(flatten_associative(remove_neutral_elements(root)))


def order_children(root: Node) -> Node:
    if isinstance(root, Lookup):
        return Lookup(tuple(sorted(root.atoms)))

    if root.arity == 0:
        return root

    children = (canonical(c) for c in root.children())
    if root.is_idempotent and (root.is_commutative and root.is_associative) :
        # Sort AND deduplicate children.
        children = sorted(set(children))
    return root.construct(children)


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
        children = _unpack(root)
    else:
        children = root.children()

    rec = (flatten_associative(c) for c in children)
    return root_type.construct(rec)


def unfuse_leaves(root: Node) -> Node:
    if isinstance(root, Lookup):
        leaves = defaultdict(set)
        for atom in root.atoms:
            leaves[atom.relative_position].add((atom.key, atom.value))

        sequence: list[Node] = [Arbitrary()] * (max(leaves.keys()) + 1)
        for offset, entries in leaves.items():
            atoms = (Atom(0, key, value) for key, value in entries)
            sequence[offset] = Lookup(tuple(atoms))
        if len(sequence) == 1:
            return sequence[0]
        else:
            return Sequence(tuple(sequence))
    else:
        children = (unfuse_leaves(c) for c in root.children())
        return root.construct(children)


def fuse_leaves(root: Node) -> Node:
    def partition_fixed_width_sequence(sequence: Iterable[Node]) -> Generator[Seq[Node]]:
        partition = []
        for element in sequence:
            if not element.has_fixed_width():
                # Not a fixed-width sequence.
                if partition:
                    # Commit partition so far and reset.
                    yield partition
                    partition = []

                # Non-fixed-width elements are their own partition.
                yield [element]

            else:
                partition.append(element)

        if partition:
            yield partition

    def fuse_partition(partition: Seq[Node]) -> Node:
        if len(partition) == 1:  # Only one item, already fused.
            return partition[0]

        for unfusable_prefix in range(len(partition)):
            if isinstance(partition[unfusable_prefix], Lookup): break
        for unfusable_suffix in reversed(range(len(partition))):
            if isinstance(partition[unfusable_suffix], Lookup): break
        unfusable_suffix += 1

        if unfusable_prefix > unfusable_suffix:  # Entire partition is unfusable.
            return Sequence(tuple(partition))

        atoms: list[Atom] = []
        remainder: list[Node] = []

        offset = 0
        for element in partition[unfusable_prefix:unfusable_suffix]:
            if isinstance(element, Lookup):
                atoms += (atom.shift(offset) for atom in element.atoms)

                for _ in range(element.width()):
                    remainder.append(Arbitrary())

            else:
                remainder.append(element)

            offset += element.width()

        fused = Lookup(atoms)

        trivial_remainder = all(isinstance(el, Arbitrary) for el in remainder)
        if not trivial_remainder:
            conjuncts = [Sequence(remainder), fused]
            fused = Conjunction(conjuncts)

        if unfusable_suffix - unfusable_prefix < len(partition):
            fused = Sequence(
                tuple(
                    partition[:unfusable_prefix] +
                    [fused] +
                    partition[unfusable_suffix:]
                )
            )

        return fused

    def fuse_conjunction(conjuncts: Iterable[Node]) -> Node:
        atoms: list[Atom] = []
        remainder: list[Node] = []

        for child in conjuncts:
            if isinstance(child, Lookup):
                atoms += child.atoms
            else:
                remainder.append(child)

        fused = Lookup(tuple(atoms))

        if not remainder:
            return fused
        else:
            remainder.append(fused)
            return Conjunction(tuple(remainder))

    if root.arity == 0:
        return root

    rec = (fuse_leaves(c) for c in root.children())
    if isinstance(root, Conjunction):
        return fuse_conjunction(rec)

    elif isinstance(root, Sequence):
        partitions = tuple(fuse_partition(partition) for partition in partition_fixed_width_sequence(rec))
        if len(partitions) == 1:
            return partitions[0]
        else:
            return Sequence(partitions)

    return root.construct(rec)
