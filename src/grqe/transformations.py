from collections import defaultdict

from grqe.query import *


def optimize(root: Node) -> Node:
    for fn in (unfuse_leaves, rewrite, fuse_leaves, share):
        root = fn(root)
    return root


def share(root: Node) -> Node:
    """Merge structurally equal nodes."""
    cached: list[Node] = []

    def rec(node: Node) -> Node:
        for cache in cached:
            # Return immediately, if structurally equal node exists.
            if cache == node:
                return cache

        # Otherwise, recurse and add to list of known nodes.
        if node.arity == 0:
            res = node
        else:
            children = (rec(c) for c in node.children())
            res = node.construct(children)
        cached.append(res)
        return res

    return rec(root)


def rewrite(node: Node) -> Node:
    """Recursively rewrite the entire given query."""
    if node.arity == 0:
        return node

    children = (rewrite(c) for c in node.children())
    node = node.construct(children)

    while True:
        # Apply rewrites until nothing changes.
        rewritten = apply_one_rewrite(node)
        if rewritten is node:
            return canonical(node)
        node = canonical(rewritten)


# Operator distribution laws, read as:
#   $key distributes over $value[0], ... $value[n]
DISTRIBUTES: dict[type, set[type]] = {
    Sequence: {Conjunction, Disjunction, Alternative},
    Conjunction: {Disjunction},
}


def apply_one_rewrite(node: Node) -> Node:
    """Non-recursive function applying re-write rules to the given node only."""
    distributing_types = DISTRIBUTES.get(type(node))
    if not distributing_types:
        # Operator does not distribute. Nothing more to do.
        return node

    for i, child in enumerate(node.children()):
        if type(child) in distributing_types:
            return distribute(node, i)
    return node


def distribute(node: Node, index: int) -> Node:
    """
    Apply distribution law to node with child at index.

    A (x1, ..., B(y1, ..., ym), ..., xn) distributes to:
    B (
        A (x1, ..., y1, ..., xn),
        ...,
        A (x1, ..., ym, ..., xn)
    )
    """
    children = list(node.children())
    promoted = children[index]

    expanded_children: list[list[Node]] = []
    for distributed_child in promoted.children():
        cur = list(children)  # Copy list of children, so subsequent mutation is only one instance.
        cur[index] = distributed_child
        expanded_children.append(cur)

    return promoted.construct(node.construct(c) for c in expanded_children)


def canonical(root: Node) -> Node:
    """
    Turns given expression to canonical form:
    - Flatten nested operators (relies on Assoc)
    - Removes unnecessary operator applications
    - Sort children (relies on Comm) while deduplicating operands (relies on Idemp + Comm, Assoc in implementation)
    - Remove neutral/absorbing elements (only Epsilon within Sequences)
    """
    for op in (
            flatten_associative,
            unpack_operators,
            order_children,
            remove_neutral_elements,
    ):
        root = op(root)
    return root


def order_children(root: Node) -> Node:
    """
    Introduces an order to (sub-)expressions.

    Also de-duplicates idempotent operators (for commutative and associative operators)
    """
    if isinstance(root, Lookup):
        return Lookup(tuple(sorted(root.atoms)))
    if isinstance(root, SpanLookup):
        return SpanLookup(root.span, tuple(sorted(root.atoms)))

    if root.arity == 0:
        return root

    children = (canonical(c) for c in root.children())
    if root.is_commutative and root.is_associative:
        if root.is_idempotent:
            children = set(children)  # De-duplicate operands.
        children = sorted(children)
    return root.construct(children)


def flatten_associative(root: Node) -> Node:
    """
    Merges nested applications of associative operators:

    A (x1, ..., A(y1, ..., ym), ..., xn) rewrites to
        A (x1, ..., y1, ..., ym, ..., xn)
    """

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


def unpack_operators(root: Node) -> Node:
    """
    Removes application of variable-arity operators to a single operand.

    A (x) rewrites to
        x
    """
    if root.arity == 0:
        return root

    root_type = type(root)
    children = [unpack_operators(child) for child in root.children()]
    if root_type.arity is None and len(children) == 1:
        return children[0]
    return root_type.construct(children)


def remove_neutral_elements(root: Node) -> Node:
    """
    Removes neutral elements from operators.

    Currently only Epsilon from Sequences.
    """
    if root.arity == 0:
        return root

    if isinstance(root, Sequence):
        children = [remove_neutral_elements(child) for child in root.children() if not isinstance(child, Epsilon)]
        match len(children):
            case 0:
                return Epsilon()
            case 1:
                return children[0]
            case _:
                return Sequence(tuple(children))

    children = (remove_neutral_elements(child) for child in root.children())
    return root.construct(children)


def unfuse_leaves(root: Node) -> Node:
    """Turns Lookup nodes into sequence of Lookups, where every Atom has offset 0."""

    if isinstance(root, Lookup):
        # Collect key-value-pairs at offset.
        leaves: dict[int, set] = defaultdict(set)
        for atom in root.atoms:
            leaves[atom.relative_position].add((atom.key, atom.value))

        # Make sequence of sufficient width using arbitrary tokens.
        # Sequence elements will be replaced with appropriate lookup.
        sequence: list[Node] = [Arbitrary()] * (max(leaves.keys()) + 1)
        for offset, entries in leaves.items():
            atoms = (Atom(0, key, value) for key, value in entries)
            sequence[offset] = Lookup(tuple(atoms))
        if len(sequence) == 1:
            return sequence[0]
        else:
            return Sequence(tuple(sequence))
    elif root.arity == 0:
        return root
    else:
        children = (unfuse_leaves(c) for c in root.children())
        return root.construct(children)


def fuse_leaves(root: Node) -> Node:
    """Finds sequences of fixed width and tries to fuse lookups within them."""

    def partition_fixed_width_sequence(sequence: Iterable[Node]) -> Generator[Seq[Node]]:
        """Partition a sequence of nodes into sub-sequences of fixed width.
        All non-fixed-width will be yielded as a singleton partition."""
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
        """Given a fixed-width partition, fuse Lookups within."""
        if len(partition) == 1:  # Only one item, already fused.
            return partition[0]

        # Find first and last Lookup node in sequence.
        # Everything before and after will just be concatenated to the fused sequence.
        for unfusable_prefix in range(len(partition)):
            if isinstance(partition[unfusable_prefix], Lookup): break
        for unfusable_suffix in reversed(range(len(partition))):
            if isinstance(partition[unfusable_suffix], Lookup): break
        unfusable_suffix += 1

        if unfusable_prefix >= unfusable_suffix:  # Entire partition is unfusable (doesn't contain Lookups).
            return Sequence(tuple(partition))

        atoms: list[Atom] = []
        remainder: list[Node] = []

        offset = 0
        for element in partition[unfusable_prefix:unfusable_suffix]:
            if isinstance(element, Lookup):
                # Collect shifted atoms
                atoms += (atom.shift(offset) for atom in element.atoms)

                # And build remainder with arbitrary according to width.
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

        if unfusable_prefix > 0 or unfusable_suffix < len(partition):
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

        if atoms:
            fused = Lookup(tuple(atoms))
            if not remainder:
                return fused
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
