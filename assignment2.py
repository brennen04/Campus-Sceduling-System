"""
This assignment assume complexity (time and space) as worst case and we don't mentioned about O(1) unless
it is worthly to mentioned
"""
from typing import List, Optional

class Edge:
    """
    Represent a directed edge in the flow network
    """
    def __init__(self, to: int, capacity: int, back: int) -> None:
        """
        initialise an edge
        """
        self.to = to
        self.capacity = capacity
        self.back = back

class NetworkFlow:
    """
    Simulate a network flow - directed graph representation for computing
    max-flow using ford-fulkerson
    """

    def __init__(self, N: int, source: int, sink: int) -> None:
        """
        Intialise a network flow

        Input:
            N (int): total number of nodes = n + m + 22 = O(n).
            source (int): source node index.
            sink (int): sink node index.
        
        Time complexity: O(n+m), where n is the number of students, m is the proposed class
        Space complexity: O(n+m), where n is the number of students, m is the proposed class
        """
        self.N = N
        self.source = source
        self.sink = sink
        self.graph = [[] for _ in range(N)]

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        """
        Add directed edge from u to v with capacity a reverse edge from v to u with 0 capacity.

        Input:
            u (int): source node index
            v (int): destination node index
            c (int): capacity of the forward edge

        Time complexity: O(1)
        Space complexity: O(1)
        """
        # forward edge
        self.graph[u].append(Edge(v, capacity, len(self.graph[v])))
        # reverse edge
        self.graph[v].append(Edge(u, 0, len(self.graph[u]) - 1))

    def _dfs(self, u: int, flow: int, is_visited: List[bool]) -> int:
        """
        Depth first search for augmenting path, return flow push, O if none

        Input:
            u (int): current node index.
            flow (int): capacity so far.
            vis (List[bool]): visited array of size N.
            
        Returns:
            int: amount of flow pushed (0 if no path found).

        Time complexity: O(N + E) = O(n+m+E), where N is the number of vertices(node) and E number of edges
        Space complexity: total O(N), = input O(N) + aux recursion stack O(N)
        """
        if u == self.sink:
            return flow
        is_visited[u] = True
        for e in self.graph[u]:
            if not is_visited[e.to] and e.capacity > 0:
                pushed = self._dfs(e.to, min(flow, e.capacity), is_visited)
                if pushed:
                    e.capacity -= pushed
                    self.graph[e.to][e.back].capacity += pushed
                    return pushed
        return 0

    def max_flow(self) -> int:
        """
        calculate the max flow from source to sink by DFS

        Time complexity: O(n(n+m))
        Space complexity: O(N) auxiliary, as for the visited list and recursion stack from dfs
        
        """
        total = 0
        is_visited = [False] * self.N
        while True:
            for i in range(self.N):
                is_visited[i] = False
            pushed = self._dfs(self.source, float('inf'), is_visited)
            if not pushed:
                break
            total += pushed
        return total

def crowdedCampus(n: int, m: int, timePreferences: List[List[int]], 
                  proposedClasses: List[List[int]], minimumSatisfaction: int) -> Optional[List[int]]:
    """
    Function desrciption:
        Allocate each of n students to one of m classes, respecting class capacities and
        ensuring at least 'minimumSatisfaction' students receive one of their top5 time slots.
    
    Approach description:
        1. Build a layered flow network:
           - source ->student nodes (capacity=1)
           - student -> timeslot nodes (capacity=1) for each preferred slot
           - timeslot -> class nodes (capacity=class_max) based on proposedClasses
           - class -> sink (capacity=class_max)
        3. Use ford fulkerson to push up to n units of flow
        4. if the max-flow < n, return None
        5. then, reconstruct each student's assignment by checking residual reverse capacities (which edges have flow)
        6. validate the assigned count whether each it is between mincapaticy and maxcapacity, 
            if fail return none
            else return the list
    
    Input:
        n: the number of students to be allocated
        m: the number of proposed classes
        timePreferences: list of list, indicate the time slot preferences
        proposedClasses: list of list, proposed time slot, the minimum and maximum number of students that can be allocated
        minimumSaticfaction: int minimum satisfaction of students get allocated within their top 5 preferred time slots

    Output:
        if feasible, return a list of length n where allocation[i] = class index for student i
        if no feasible allocation, return None

    Time complexity:
        O(n(n+m)), where n is the number of students, m is the proposed class

    Time complexity analysis: (n is the number of students,
                                m is the proposed class)
        - graph construction: O(n + m)
        - max-flow (ford-fulkerson with dfs): O(n) * O(n+m)
        - reconstruction: O(n*m)
        - final validation: O(n+m)
        thus total: O(n(n+m))
    
    Aux space complexity:
        O(n+m)
        - same graph construction O(n+m)
        - recursion stack O(N)
        - assignment list of length n


    Total space complexity (input + aux):
        O(n+m), from input+aux
    """     
    
    NUM_SLOTS = 20
    source =  0
    student_index =  1
    timeslot_index = student_index + n
    class_index = timeslot_index + NUM_SLOTS
    sink = class_index + m
    total_nodes = sink + 1

    network = NetworkFlow(total_nodes, source, sink)

    # source -> student
    for i in range(n):
        network.add_edge(source, student_index + i, 1)

    # student -> timeslot
    for i in range(n):
        stu: int = student_index + i
        for slot in timePreferences[i]:
            network.add_edge(stu, timeslot_index + slot, 1)

    # timeslot -> class
    for j in range(m):
        ts, minimum, maximum = proposedClasses[j]
        network.add_edge(timeslot_index + ts, class_index + j, maximum)

    # class -> sink
    for j in range(m):
        _, minimum, maximum = proposedClasses[j]
        network.add_edge(class_index + j, sink, maximum)

    flow = network.max_flow()
    if flow != n:
        return None
    
    #Reconstruct assignments
    assignment = [-1] * n
    counts = [0] * m
    satisfaction = 0

    for i in range(n):
        stu = student_index + i
        chosen = -1
        
        for slot in range(NUM_SLOTS):  # O(20)=O(1)
            for e in network.graph[timeslot_index + slot]:
                if e.to == stu and e.capacity > 0:
                    chosen = slot
                    e.capacity -= 1
                    break
            if chosen >= 0:
                break
        if chosen < 0:
            return None
        
        for j in range(m):  # O(n)
            for e in network.graph[class_index + j]:
                if e.to == timeslot_index + chosen and e.capacity > 0:
                    assignment[i] = j
                    e.capacity -= 1
                    break
            if assignment[i] >= 0:
                break
        if assignment[i] < 0:
            return None
        counts[assignment[i]] += 1
        for k in range(5):  # O(1)
            if timePreferences[i][k] == chosen:
                satisfaction += 1
                break

    # verify capacities: O(n)
    for j in range(m):
        mn, mx = proposedClasses[j][1], proposedClasses[j][2]
        if not (mn <= counts[j] <= mx):
            return None
        
    # verify satisfaction
    if satisfaction < minimumSatisfaction:
        return None
    return assignment

class Node:
    def __init__(self):
        """
        Create a node which carries list of size 26 and a boolean value
        Time complexity: O(1) per node
        Space complexity: O(1) auxiliary space per node
        """
        self.children = [None] * 26
        self.is_terminal = False 

class Bad_AI:
    def __init__(self, list_words: List[str]):
        """
        Function desrciption:
            Build a prefix-trie of all based on the input list in O(C) time and space, where C is the the total
            characters in list_words
        
        Approach description:
            Insert each character of each word into the trie, last character of a word is marked as true
        
        Input:
            list_words: list of N unique lowercase words, total C characters
        
        Time complexity: 
            O(C), where C is the total characters in list_words
        
        Time complexity analysis:
            As we scan each word once, O(N) and allocate each character of the word to the trie, O(C),
            thus total = O(N) + O(C) = O(C) 

        Space complexity:
            O(C) auxliary
        
        Space complexity analysis:
            we create one node per character with no shared prefix (worst case). Each node has a fixed 26 slot array and 
            a bolean value. Thus total space O(C)

        """
        self.root = Node()
        self.max_length = 0

        for word in list_words:
            word_length = len(word)
            if word_length > self.max_length:
                self.max_length = word_length
            node = self.root
            for char in word:
                i = ord(char) - ord('a')
                if node.children[i] is None:
                    node.children[i] = Node()
                node = node.children[i]
            node.is_terminal = True

    def check_word(self, sus_word: str) -> List[str]:
        """
        Function description:
            return a list of words that has a Levenshtein distances to sus_word of 1, through substituion only
        
        Approach description:
            for each postion i in sus_word, we try to substitute each of the other 25 letters. For each variant, we walk from the
            trie root following sus_word[j] for j=/1 and the substitue at j==1. If we succesfully reach the terminal node (last char),
            we output that word.
        
        Input:
            sus_word: a string of length J

        Output:
            A list of words that is has a Levenshtein distances to sus_word of 1 through substituion only

        Time complexity:
            O(J^2 + X), where J is the length of sus_word and X is the total characters returned in the correct result

        Time complexity analysis:
            - First loop over J times, O(J)
            - Second loop over 0...25, constant 25, so O(1)
            - Third loop, exact-match trie walk of length J each time, O(J)
            - Building each mathc via join() cost O(J) per match, summing to O(X)
            Total = O(J^2 + X)
            If N > J then our solution is within the bound of O(J * N + X),
            but is J > N then our solution exceeds the bound of O(J * N + X)
        
        Space complexity: 
            total O(J + X), O(X) is auxiliary, O(J) input, where J is the length of sus_word and X is the total characters returned in the correct result
            

        Space complexity analysis:
            Input string length J and temporary string (final_word) length J, O(J)
            Result list of X characters, O(X)
        """

        sus_length = len(sus_word)
        if sus_length > self.max_length:
            return  []
        
        results = []
        root = self.root
        APLHABET = "abcdefjghijklmnopqrstuvwxyz"

        for i in range(sus_length): # O(J) time
            sus_char = sus_word[i]
            for letter in APLHABET: # O(1) time as we know there is only lowercase letters
                if letter == sus_char:
                    continue

                node = root
                is_match = True

                # Exact-match walk of the one-off variant
                for j in range(sus_length): # O(J) time
                    if j == i:
                        char = letter
                    else:
                        char = sus_word[j]
                    index = ord(char) - ord('a')
                    child = node.children[index]
                    if child is None:
                        is_match = False
                        break
                    node = child
                
                # get the final word if we land on the terminal node
                if node.is_terminal and is_match:
                    characters = []
                    for k in range(sus_length): # O(J) time

                         # Build in one pass of length J (counts toward X)
                        if k == i:
                            characters.append(letter)
                        else:
                            characters.append(sus_word[k])
                    final_word = "".join(characters) # O(J) time
                    results.append(final_word)
        return results # O(X) space
                

        
if __name__ == "__main__":
    list_words = ["aaa", "abc", "xyz", "aba", "aaaa"]
    list_sus = ["aaa", "axa", "ab", "xxx", "aaab"]
    my_ai = Bad_AI(list_words)
    for sus_word in list_sus:
       my_answer = my_ai.check_word(sus_word)
       print(my_answer)