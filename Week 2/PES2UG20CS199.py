'''
UE20CS302 (D Section)
Machine Intelligence 
Week 2: Search Algorithms

Mitul Joby
PES2UG20CS199
'''

from queue import LifoQueue, PriorityQueue
from copy import deepcopy

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    
    n = len(heuristic)
    visited = [0 for i in range(n)]
    nodeQueue = PriorityQueue()
    nodeQueue.put((heuristic[start_point], [start_point], start_point, 0))

    while(nodeQueue.qsize() != 0):
        estimate, nodePath , currentNode , nodeCost = nodeQueue.get()

        if visited[currentNode] == 0:
            visited[currentNode] = 1

            if currentNode in goals:
                return nodePath

            for nextNode in range(1, n):
                if cost[currentNode][nextNode] > 0 and visited[nextNode] == 0:
                    totalCost = nodeCost + cost[currentNode][nextNode]
                    estimatedCost = totalCost + heuristic[nextNode]
                    pathToNextNode = deepcopy(nodePath)
                    pathToNextNode.append(nextNode)
                    nodeQueue.put((estimatedCost, pathToNextNode, nextNode, totalCost))

    return []


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    n = len(cost)                               
    visited = [0 for i in range(n)]
    stack = LifoQueue()           
    stack.put((start_point, [start_point]))

    while(stack.qsize() != 0):
        node, nodePath = stack.get()

        if visited[node] == 0:
            visited[node] = 1

            if node in goals:
                return nodePath

            for nextNode in range(n-1, 0, -1):
                if cost[node][nextNode] > 0 and visited[nextNode] == 0:
                        pathTillNextNode = deepcopy(nodePath)
                        pathTillNextNode.append(nextNode)
                        stack.put((nextNode, pathTillNextNode))

    return []
