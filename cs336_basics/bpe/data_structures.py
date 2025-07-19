from typing import Any, Iterator, Optional


class Node:
    """
    A node in a doubly linked list.
    """
    __slots__ = ('value', 'prev', 'next')

    def __init__(self, value: Any):
        self.value: Any = value
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

    def __repr__(self) -> str:
        return f"Node({self.value!r})"


class DoublyLinkedList:
    """
    A simple doubly linked list supporting append, removal, and insertion.
    """
    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.length: int = 0

    def append(self, value: Any) -> Node:
        """Append a new node with the given value at the end of the list."""
        node = Node(value)
        if self.head is None:
            self.head = self.tail = node
        else:
            assert self.tail is not None  # for type checker
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        self.length += 1
        return node

    def remove(self, node: Node) -> None:
        """Remove the given node from the list."""
        if node.prev:
            node.prev.next = node.next
        else:
            # Removing head
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            # Removing tail
            self.tail = node.prev
        node.prev = node.next = None
        self.length -= 1

    def insert_after(self, node: Node, value: Any) -> Node:
        """Insert a new node with value immediately after the given node."""
        new_node = Node(value)
        nxt = node.next
        node.next = new_node
        new_node.prev = node
        new_node.next = nxt
        if nxt:
            nxt.prev = new_node
        else:
            # Inserted at the end
            self.tail = new_node
        self.length += 1
        return new_node

    def __iter__(self) -> Iterator[Node]:
        """Iterate over nodes from head to tail."""
        current = self.head
        while current:
            yield current
            current = current.next

    def __reversed__(self) -> Iterator[Node]:
        """Iterate over nodes from tail to head."""
        current = self.tail
        while current:
            yield current
            current = current.prev

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        values = [repr(node.value) for node in self]
        return f"DoublyLinkedList([{', '.join(values)}])"
