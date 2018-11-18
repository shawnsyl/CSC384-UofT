from csp import Constraint, Variable, CSP
import random
import util

class UnassignedVars:
    '''class for holding the unassigned variables of a CSP. We can extract
       from, re-initialize it, and return variables to it.  Object is
       initialized by passing a select_criteria (to determine the
       order variables are extracted) and the CSP object.

       select_criteria = ['random', 'fixed', 'mrv'] with
       'random' == select a random unassigned variable
       'fixed'  == follow the ordering of the CSP variables (i.e.,
                   csp.variables()[0] before csp.variables()[1]
       'mrv'    == select the variable with minimum values in its current domain
                   break ties by the ordering in the CSP variables.
    '''
    def __init__(self, select_criteria, csp):
        if select_criteria not in ['random', 'fixed', 'mrv']:
            print "Error UnassignedVars given an illegal selection criteria {}. Must be one of 'random', 'stack', 'queue', or 'mrv'".format(select_criteria)
        self.unassigned = list(csp.variables())
        self.csp = csp
        self._select = select_criteria
        if select_criteria == 'fixed':
            #reverse unassigned list so that we can add and extract from the back
            self.unassigned.reverse()

    def extract(self):
        if not self.unassigned:
            print "Warning, extracting from empty unassigned list"
            return None
        if self._select == 'random':
            i = random.randint(0,len(self.unassigned)-1)
            nxtvar = self.unassigned[i]
            self.unassigned[i] = self.unassigned[-1]
            self.unassigned.pop()
            return nxtvar
        if self._select == 'fixed':
            return self.unassigned.pop()
        if self._select == 'mrv':
            nxtvar = min(self.unassigned, key=lambda v: v.curDomainSize())
            self.unassigned.remove(nxtvar)
            return nxtvar

    def empty(self):
        return len(self.unassigned) == 0

    def insert(self, var):
        if not var in self.csp.variables():
            print "Error, trying to insert variable {} in unassigned that is not in the CSP problem".format(var.name())
        else:
            self.unassigned.append(var)

def bt_search(algo, csp, variableHeuristic, allSolutions, trace):
    '''Main interface routine for calling different forms of backtracking search
       algorithm is one of ['BT', 'FC', 'GAC']
       csp is a CSP object specifying the csp problem to solve
       variableHeuristic is one of ['random', 'fixed', 'mrv']
       allSolutions True or False. True means we want to find all solutions.
       trace True of False. True means turn on tracing of the algorithm

       bt_search returns a list of solutions. Each solution is itself a list
       of pairs (var, value). Where var is a Variable object, and value is
       a value from its domain.
    '''
    varHeuristics = ['random', 'fixed', 'mrv']
    algorithms = ['BT', 'FC', 'GAC']

    #statistics
    bt_search.nodesExplored = 0

    if variableHeuristic not in varHeuristics:
        print "Error. Unknown variable heursitics {}. Must be one of {}.".format(
            variableHeuristic, varHeuristics)
    if algo not in algorithms:
        print "Error. Unknown algorithm heursitics {}. Must be one of {}.".format(
            algo, algorithms)

    uv = UnassignedVars(variableHeuristic,csp)
    Variable.clearUndoDict()
    for v in csp.variables():
        v.reset()
    if algo == 'BT':
         solutions = BT(uv, csp, allSolutions, trace)
    elif algo == 'FC':
        for cnstr in csp.constraints():
            if cnstr.arity() == 1:
                FCCheck(c, None, None)  #FC with unary constraints at the root
        solutions = FC(uv, csp, allSolutions, trace)
    elif algo == 'GAC':
        GacEnforce(csp.constraints(), csp, None, None) #GAC at the root
        solutions = GAC(uv, csp, allSolutions, trace)

    return solutions, bt_search.nodesExplored

def BT(unAssignedVars, csp, allSolutions, trace):
    '''Backtracking Search. unAssignedVars is the current set of
       unassigned variables.  csp is the csp problem, allSolutions is
       True if you want all solutionss trace if you want some tracing
       of variable assignments tried and constraints failed. Returns
       the set of solutions found.

      To handle finding 'allSolutions', at every stage we collect
      up the solutions returned by the recursive  calls, and
      then return a list of all of them.

      If we are only looking for one solution we stop trying
      further values of the variable currently being tried as
      soon as one of the recursive calls returns some solutions.

      if trace:
    '''
    if unAssignedVars.empty():
        if trace: print "{} Solution Found".format(csp.name())
        soln = []
        for v in csp.variables():
            soln.append((v, v.getValue()))
        return [soln]  #each call returns a list of solutions found
    bt_search.nodesExplored += 1
    solns = []         #so far we have no solutions recursive calls
    nxtvar = unAssignedVars.extract()
    if trace: print "==>Trying {}".format(nxtvar.name())
    for val in nxtvar.domain():
        if trace: print "==> {} = {}".format(nxtvar.name(), val)
        nxtvar.setValue(val)
        constraintsOK = True
        for cnstr in csp.constraintsOf(nxtvar):
            if cnstr.numUnassigned() == 0:
                if not cnstr.check():
                    constraintsOK = False
                    if trace: print "<==falsified constraint\n"
                    break
        if constraintsOK:
            new_solns = BT(unAssignedVars, csp, allSolutions, trace)
            if new_solns:
                solns.extend(new_solns)
            if len(solns) > 0 and not allSolutions:
                break #don't bother with other values of nxtvar
                      #as we found a soln.
    nxtvar.unAssign()
    unAssignedVars.insert(nxtvar)
    return solns

def FCCheck(cnstr, reasonVar, reasonVal):
    if cnstr.numUnassigned() != 1:
        if trace: print "Error FCCheck called on constraint {} with {} neq 1 unassigned vars".format(cnstr.name(), cnstr.numUnassignedVars)
    var = cnstr.unAssignedVars()[0]
    for val in var.curDomain():
        var.setValue(val)
        if not cnstr.check():
            var.pruneValue(val, reasonVar, reasonVal)
        var.unAssign()  #NOTE WE MUST UNDO TRIAL ASSIGNMENT
    if var.curDomainSize() == 0:
        return "DWO"
    return "OK"

def FC(unAssignedVars, csp, allSolutions, trace):
    '''Forward checking search.
       unAssignedVars is the current set of
       unassigned variables.  csp is the csp
       problem, allSolutions is True if you want all solutionsl trace
       if you want some tracing of variable assignments tried and
       constraints failed.

       RETURNS LIST OF ALL SOLUTIONS FOUND.

       Finding allSolutions is handled just as it was in BT.  Except
       that when we are not looking for all solutions and we stop
       early because one of the recursive calls found a solution we
       must make sure that we restore all pruned values before
       returning.
    '''
    if unAssignedVars.empty():
        if trace: print "{} Solution Found".format(csp.name())
        soln = []
        for v in csp.variables():
            soln.append((v, v.getValue()))
        return [soln]  # each call returns a list of solutions found
    bt_search.nodesExplored += 1
    solns = []  # so far we have no solutions recursive calls
    nxtvar = unAssignedVars.extract()
    if trace: print "==>Trying {}".format(nxtvar.name())
    for val in nxtvar.curDomain():
        if trace: print "==> {} = {}".format(nxtvar.name(), val)
        nxtvar.setValue(val)
        noDWO = True
        for cnstr in csp.constraintsOf(nxtvar):
            if cnstr.numUnassigned() == 1:
                if FCCheck(cnstr, nxtvar, val) == "DWO":
                    noDWO = False
                    if trace: print "<==Domain Wipe Out\n"
                    break
        if noDWO:
            new_solns = FC(unAssignedVars, csp, allSolutions, trace)
            if new_solns:
                solns.extend(new_solns)
            if len(solns) > 0 and not allSolutions:
                Variable.restoreValues(nxtvar, val)
                break  # don't bother with other values of nxtvar
                # as we found a soln.
        Variable.restoreValues(nxtvar, val)
    nxtvar.unAssign()
    unAssignedVars.insert(nxtvar)
    return solns

def GacEnforce(constraints, csp, reasonVar, reasonVal):
    '''Establish GAC on constraints by pruning values
       from the current domains of the variables.
       Return "OK" if completed "DWO" if found
       a domain wipe out.'''
    #your implementation for Question 3 goes in this function body
    #you must not change the function parameters
    #ensure that you return one of "OK" or "DWO"
    while constraints:
        cnstr = constraints.pop()
        for var in cnstr.unAssignedVars():
            for val in var.curDomain():
                if not cnstr.hasSupport(var, val):
                    var.pruneValue(val, reasonVar, reasonVal)
                    if var.curDomainSize() == 0:
                        return "DWO"
                    for recheck in csp.constraintsOf(var):
                        if recheck != cnstr and not recheck in constraints:
                            constraints.append(recheck)
    return "OK"


def GAC(unAssignedVars, csp, allSolutions, trace):
    '''GAC search.
       unAssignedVars is the current set of
       unassigned variables.  csp is the csp
       problem, allSolutions is True if you want all solutionsl trace
       if you want some tracing of variable assignments tried and
       constraints failed.

       RETURNS LIST OF ALL SOLUTIONS FOUND.

       Finding allSolutions is handled just as it was in BT.  Except
       that when we are not looking for all solutions and we stop
       early because one of the recursive calls found a solution we
       must make sure that we restore all pruned values before
       returning.
    '''
    if unAssignedVars.empty():
        if trace: print "{} Solution Found".format(csp.name())
        soln = []
        for v in csp.variables():
            soln.append((v, v.getValue()))
        return [soln]  # each call returns a list of solutions found
    bt_search.nodesExplored += 1
    solns = []  # so far we have no solutions recursive calls
    nxtvar = unAssignedVars.extract()
    if trace: print "==>Trying {}".format(nxtvar.name())
    for val in nxtvar.curDomain():
        if trace: print "==> {} = {}".format(nxtvar.name(), val)
        nxtvar.setValue(val)
        noDWO = True
        if GacEnforce(csp.constraintsOf(nxtvar), csp, nxtvar, val) == "DWO":
            noDWO = False
        if noDWO:
            new_solns = GAC(unAssignedVars, csp, allSolutions, trace)
            if new_solns:
                solns.extend(new_solns)
            if len(solns) > 0 and not allSolutions:
                Variable.restoreValues(nxtvar, val)
                break  # don't bother with other values of nxtvar
                # as we found a soln.
        Variable.restoreValues(nxtvar, val)
    nxtvar.unAssign()
    unAssignedVars.insert(nxtvar)
    return solns