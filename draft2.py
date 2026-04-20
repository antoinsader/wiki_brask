class DynamicArray:
    
    def __init__(self, capacity: int):
        self.ar = [None for i in range(capacity)]
        self.size = 0
        self.capacity = capacity

    def get(self, i: int) -> int:
        return self.ar[i]

    def set(self, i: int, n: int) -> None:
        self.ar[i] = n
        
    def pushback(self, n: int) -> None:
        if self.size  == self.capacity:
            self.resize()

        self.ar[self.size] = n
        self.size += 1



    def popback(self) -> int:
        p = self.ar[self.size - 1]
        self.ar = self.ar[:self.size - 1]
        self.size  -= 1
        return p
 

    def resize(self) -> None:
        self.ar = self.ar + [None for _ in range(self.capacity)]
        self.capacity = self.capacity * 2


    def getSize(self) -> int:
        return self.size        
    
    def getCapacity(self) -> int:
        return self.capacity