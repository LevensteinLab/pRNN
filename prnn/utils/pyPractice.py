class Node: 
    def __init__(self,data):
        self.data = data 
        self.next = None

class LinkedList: 
    def __init__(self):
        self.head = None

    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def display(self):
        current = self.head
        while current: 
            print(current.data, end=" -> ")
            current = current.next
        print("None")

l1 = LinkedList()
l1.append(10)
l1.append(20)
l1.append(30)
l1.display()

class Queue: 
    def __init__(self):
        self.front = None
        self.rear = None
    
    def enqueue(self,data):
        new_node = Node(data)
        if self.rear is None:
            self.rear = new_node
            return 
        self.rear.next = new_node
        self.rear = new_node

    def dequeue(self):
        if self.front is None:
            print("Queue is empty")
            return None
        dequeued_data = self.front.data
        self.front = self.front.next()
        if self.front is None:
            self.rear = None
        return dequeued_data

class Stack:
    def __init__(self):
        self.top = None
    
    def push(self, data):
        new_node = Node(data)
        new_node.next = self.top
        self.top = new_node

    def pop(self):
        if self.top is None: 
            print("stack is empty")
            return None
        popped_data = self.top.data
        self.top = self.top.next
        return popped_data
    
l2 = Queue()
l2.enqueue(14)
l2.enqueue(13)
l2.dequeue()

l3 = Stack()
l3.push(2)
l3.push(4)
l3.pop()
l3.push(7)

