# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    visited = []
    paths = util.Stack()
    # push start state, direction and cost into stack
    """
    [(state, dir, cost), (state, dir, cost), (state, dir, cost)]
    [(state, dir, cost), (state, dir, cost), (state, dir, cost)]
    """
    paths.push([(problem.getStartState(), 'Stop', 0)])

    while not paths.isEmpty():
        curPath = paths.pop()
        """
                print curPath[-1][0] # state
                print curPath[-1][1] # dir
                print curPath[-1][2] # cost
        """
        end = curPath[-1]
        if not end[0] in visited: # if we haven't visited this node before
            visited += [end[0]] # add it to visited list
            if problem.isGoalState(end[0]): # check if goal
                return [i[1] for i in curPath[1:]] # return directions
            successors = problem.getSuccessors(end[0]) # get all successors
            notvisited = [s for s in successors if not s[0] in visited] # then see if the successors have been visited
            for s in notvisited: # for all successors we haven't visited push into stack
                newcost = s[2] + end[2]
                newPath = list(curPath)
                newPath += [(s[0], s[1], newcost)]
                paths.push(newPath)
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    visited = []
    paths = util.Queue()
    # push start state, direction and cost into Queue
    paths.push([(problem.getStartState(), 'Stop', 0)])

    while not paths.isEmpty():
        curPath = paths.pop()
        end = curPath[-1]
        if not end[0] in visited:
            visited += [end[0]]
            if problem.isGoalState(end[0]):
                return [i[1] for i in curPath[1:]]
            successors = problem.getSuccessors(end[0])
            notvisited = [s for s in successors if not s[0] in visited]
            for s in notvisited:
                newcost = s[2] + end[2]
                newPath = list(curPath)
                newPath += [(s[0], s[1], newcost)]
                paths.push(newPath)

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visited = []
    paths = util.PriorityQueue()
    # push start state, direction and cost into Queue
    paths.push([(problem.getStartState(), 'Stop', 0)], 0)

    while not paths.isEmpty():
        curPath = paths.pop()
        end = curPath[-1]
        if not end[0] in visited:
            visited += [end[0]]
            if problem.isGoalState(end[0]):
                return [i[1] for i in curPath[1:]]
            successors = problem.getSuccessors(end[0])
            notvisited = [s for s in successors if not s[0] in visited]
            for s in notvisited:
                newcost = s[2] + end[2]
                newPath = list(curPath)
                newPath += [(s[0], s[1], newcost)]
                paths.push(newPath, newcost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visited = []
    paths = util.PriorityQueue()
    # push start state, direction and cost into Queue
    # paths format:
    # pqueue of path, f as priority([list of states], f value) (f = g + h)
    # state format:
    # (gameState, direction, g, h, f)
    paths.push([(problem.getStartState(), 'Stop', 0, 0, 0)], 0)

    while not paths.isEmpty():
        pathCheck = paths.pop()
        # tie-breaking:

        fThreshhold = pathCheck[-1][4] # the lowest f value we got so far since it was popped first
        pathCheckList = []
        pathCheckList += [pathCheck] # add first path popped to the examine-list

        while True and not paths.isEmpty():
            pathCheck = paths.pop()
            # check what f-value it has
            # if pathCheck has a higher f value, then we don't need to keep looking
            # else, add this pathCheck to a list of pathChecks so we can evaluate the g
            if pathCheck[-1][4] > fThreshhold:
                paths.push(pathCheck, pathCheck[-1][4]) # push pathCheck back into priorityQueue
                                                        # since it has a diff f value
                break
            else:
                pathCheckList += [pathCheck] # add to list of paths to be checked

        chosenOne = find_highest_g(pathCheckList) # out of these, find the path with the highest g value

        if chosenOne != None:
            pathCheckList.remove(chosenOne) # remove the chosenOne from pathCheckList
                                            # since everything in pathCheckList gets put back in Queue

        for k in pathCheckList:
            paths.push(k, k[-1][4]) # put everything in pathCheckList back into the priorityQueue

        # tie-breaking done, do the actual expansion
        end = chosenOne[-1]
        if not end[0] in visited:
            visited += [end[0]]
            if problem.isGoalState(end[0]):
                return [i[1] for i in chosenOne[1:]]
            successors = problem.getSuccessors(end[0])
            notvisited = [s for s in successors if not s[0] in visited]
            for s in notvisited:
                gcost = s[2] + end[2] #g
                hcost = heuristic(s[0], problem) #h
                fcost = gcost + hcost #f
                newPath = list(chosenOne)
                newPath += [(s[0], s[1], gcost, hcost, fcost)]
                paths.push(newPath, fcost)
    return []

def find_highest_g(pathCheckList):
    if len(pathCheckList) == 0:
        return None
    result = pathCheckList[0]
    highg = result[-1][2]
    for p in pathCheckList[1:]:
        result = p if p[-1][2] > highg else result
        highg = p[-1][2] if p[-1][2] > highg else highg
    return result
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
