import csp_problems
import backtracking
import argparse

#Class to structure the data of a single instance of the plane scheduling problem
#test sequence--one solution (only one legal sequencing)
p1 = csp_problems.PlaneProblem(
    #planes
    ['AC-1'],
    #the flights
    ['AC01', 'AC02', 'AC03', 'AC04', 'AC05', 'AC06'],
    #flights each plane can fly
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05', 'AC06']],
    #flights each plane can start with
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05', 'AC06']],
    #Flights that can follow each other
    [('AC01', 'AC02'), ('AC02', 'AC03'),
     ('AC03', 'AC04'), ('AC04', 'AC05'),
     ('AC05', 'AC06')],
    #maintenance depots at end of these flights
    [],
    #min maintenance frequency
    7)

#test sequence--two solutions
p2 = csp_problems.PlaneProblem(
    #planes
    ['AC-1'],
    #the flights
    ['AC01', 'AC02', 'AC03', 'AC04', 'AC05', 'AC06'],
    #flights each plane can fly
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05', 'AC06']],
    #flights each plane can start with
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05', 'AC06']],
    #Flights that can follow each other
    [('AC01', 'AC02'), ('AC02', 'AC03'),
     ('AC03', 'AC04'), ('AC04', 'AC05'),
     ('AC05', 'AC06'), ('AC04', 'AC06'), ('AC06', 'AC05')],
    #maintenance depots at end of these flights
    [],
    #min maintenance frequency
    7)

#test maintenance frequence---24 solutions all with 'AC03' in the middle
p3 = csp_problems.PlaneProblem(
    #planes
    ['AC-1'],
    #the flights
    ['AC01', 'AC02', 'AC03', 'AC04', 'AC05'],
    #flights each plane can fly
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05']],
    #flights each plane can start with
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05']],
    #Flights that can follow each other
    [('AC01', 'AC02'), ('AC01', 'AC03'),  ('AC01', 'AC04'), ('AC01', 'AC05'),
     ('AC02', 'AC01'), ('AC02', 'AC03'),  ('AC02', 'AC04'), ('AC02', 'AC05'),
     ('AC03', 'AC01'), ('AC03', 'AC02'),  ('AC03', 'AC04'), ('AC03', 'AC05'),
     ('AC04', 'AC01'), ('AC04', 'AC02'),  ('AC04', 'AC03'), ('AC04', 'AC05'),
     ('AC05', 'AC01'), ('AC05', 'AC02'),  ('AC05', 'AC03'), ('AC05', 'AC04')],
    #maintenance depots at end of these flights
    ['AC03'],
    #min maintenance frequency
    3)

#test maintenance frequence---zero solutions
p4 = csp_problems.PlaneProblem(
    #planes
    ['AC-1'],
    #the flights
    ['AC01', 'AC02', 'AC03', 'AC04', 'AC05'],
    #flights each plane can fly
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05']],
    #flights each plane can start with
    [['AC-1', 'AC01', 'AC02', 'AC03', 'AC04', 'AC05']],
    #Flights that can follow each other
    [('AC01', 'AC02'), ('AC01', 'AC03'),  ('AC01', 'AC04'), ('AC01', 'AC05'),
     ('AC02', 'AC01'), ('AC02', 'AC03'),  ('AC02', 'AC04'), ('AC02', 'AC05'),
     ('AC03', 'AC01'), ('AC03', 'AC02'),  ('AC03', 'AC04'), ('AC03', 'AC05'),
     ('AC04', 'AC01'), ('AC04', 'AC02'),  ('AC04', 'AC03'), ('AC04', 'AC05'),
     ('AC05', 'AC01'), ('AC05', 'AC02'),  ('AC05', 'AC03'), ('AC05', 'AC04')],
    #maintenance depots at end of these flights
    ['AC03'],
    #min maintenance frequency
    2)

#ordinary test problems
p5 = csp_problems.PlaneProblem(
    #planes
    ['AC-1', 'AC-2', 'AC-3'],
    #the flights
    ['AC98', 'AC99', 'AC100', 'AC101', 'AC200', 'AC201'],
    #flights each plane can fly
    [['AC-1', 'AC98', 'AC99'],
     ['AC-2', 'AC100', 'AC101'],
     ['AC-3', 'AC200', 'AC201']],
    #flights each plane can start with
    [['AC-1', 'AC98'],
     ['AC-2', 'AC100'],
     ['AC-3', 'AC200']],
    #Flights that can follow each other
    [('AC98', 'AC99'), ('AC99', 'AC98'),
     ('AC100', 'AC101'), ('AC101', 'AC100'),
     ('AC200', 'AC201'), ('AC201', 'AC200')],
    #maintenance depots at end of these flights
    ['AC98', 'AC101', 'AC200'],
    #min maintenance frequency
    2)

p6 = csp_problems.PlaneProblem(
    #planes
    ['AC-1', 'AC-2', 'AC-3'],
    #the flights
    ['AC001', 'AC002', 'AC003', 'AC004'],
    #flights each plane can fly
    [['AC-1', 'AC001', 'AC002', 'AC003', 'AC004'],
     ['AC-2', 'AC001', 'AC002', 'AC003', 'AC004'],
     ['AC-3', 'AC001', 'AC002', 'AC003', 'AC004']],
    #flights each plane can start with
    [['AC-1', 'AC001'], ['AC-2', 'AC001'], ['AC-3', 'AC004']],
    #Flights that can follow each other
    [('AC001', 'AC002'), ('AC002', 'AC001'),
     ('AC002', 'AC003'), ('AC003', 'AC002'),
     ('AC003', 'AC004'), ('AC004', 'AC003')],
    #maintenance depots at end of these flights
    ['AC001', 'AC002', 'AC003', 'AC004'],
    #min maintenance frequency
    2)

#harder problem---but should still be solvable.
p7 = csp_problems.PlaneProblem(
    #planes
    ['AC-1', 'AC-2', 'AC-3', 'BA-1', 'BA-2'], #the planes
    #the flights
    ['AC001', 'AC002', 'AC003', 'AC004', 'AC005', 'AC006', 'AC007',
     'BA001', 'BA002', 'BA003', 'BA004', 'BA005', 'BA006'],
    #flights each plane can fly
    [['AC-1', 'AC001', 'AC002', 'AC003'],
     ['AC-2', 'AC004'],
     ['AC-3', 'AC005', 'AC006', 'AC007'],
     ['BA-1', 'BA001', 'BA002', 'BA003'],
     ['BA-2', 'BA004', 'BA005', 'BA006']],
    #flights each plane can start with
    [['AC-1', 'AC001', 'AC003', 'AC004'],
     ['AC-2', 'AC001', 'AC003', 'AC004'],
     ['AC-3', 'AC004', 'AC006', 'AC007'],
     ['BA-1', 'BA001', 'BA003', 'BA004', 'BA006'],
     ['BA-2', 'BA001', 'BA003', 'BA004', 'BA006']],
    #Flights that can follow each other
    [('AC001', 'AC002'), ('AC002', 'AC001'),
     ('AC001', 'AC003'), ('AC003', 'AC001'), ('AC001', 'AC004'), ('AC004', 'AC001'),
     ('AC002', 'AC003'), ('AC003', 'AC002'), ('AC003', 'AC004'), ('AC004', 'AC003'),
     ('AC004', 'AC007'), ('AC007', 'AC004'), ('AC004', 'AC006'), ('AC006', 'AC004'),
     ('AC005', 'AC006'), ('AC006', 'AC005'), ('AC005', 'AC007'), ('AC007', 'AC005'),
     ('AC006', 'AC007'), ('AC007', 'AC006'),
     ('BA001', 'BA002'), ('BA002', 'BA003'), ('BA001', 'BA003'), ('BA003', 'BA001'),
     ('BA001', 'BA006'), ('BA006', 'BA001'), ('BA001', 'BA004'), ('BA004', 'BA001'),
     ('BA002', 'BA003'), ('BA003', 'BA002'), ('BA003', 'BA004'), ('BA004', 'BA003'),
     ('BA003', 'BA006'), ('BA006', 'BA003'), ('BA004', 'BA005'), ('BA005', 'BA004'),
     ('BA004', 'BA006'), ('BA006', 'BA004'), ('BA005', 'BA006'), ('BA006', 'BA005')],
    #maintenance depots at end of these flights
    ['AC001', 'AC003', 'AC004', 'AC007', 'AC006',
     'BA001', 'BA003', 'BA006', 'BA004'],
    #Min maintenance frequency
    3)

problems = [p1,p2,p3,p4,p5,p6,p7]

def check_plane_solution(pp, s):
    '''Check is solution is a valid solution to plane problem pp.
       Returned format of solution must be a list of lists with one
       list for each plane. Each list L starts with the plane (L[0])
       and the following elements of L are the flights that this plane
       flys in the sequence flown. E.g., if solution is
       [['AC-1', 'AC01', 'AC02'], ['AC-2', 'AC03', 'AC04'], ['AC-3']]
       then there are three planes AC-1', 'AC-2', and 'AC-3'. AC-1 first
       flies flight AC01 then AC02. AC-2 first flys AC03 then AC04, and
       AC-3 does not fly any flights'''
    if not check_valid_planes(pp,s):
        return False
    if not check_valid_flights(pp,s):
        return False
    val = check_can_fly(pp, s)
    val = check_sequencing(pp, s) and val
    val = check_maintenance(pp, s) and val
    val = check_all_flights_scheduled_once(pp, s) and val
    return val

def check_valid_planes(pp,s):
    for i,pl in enumerate(s):
        if len(pl) == 0:
            print "Error solution invalid, no plane specified in the {}-th list".format(i+1)
            return False
        if pl[0] not in pp.planes:
            print "Error solution invalid, non-existent plane {} in the {}-th list".format(pl[0],i+1)
            return False
    return True

def check_valid_flights(pp,s):
    for i,pl in enumerate(s):
        if len(pl) == 0:
            return False
        for f in pl[1:]:
            if f not in pp.flights:
                print "Error solution invalid, non-existent flight {} in the {}-th list".format(f, i+1)
                return False
    return True

def check_can_fly(pp, s):
    for i, pl in enumerate(s):
        plane = pl[0]
        flights = pl[1:]
        for j, f in enumerate(flights):
            if j == 0:
                if not f in pp.can_start(plane):
                    print "Error solution invalid, plane {} starts with invalid flight {}".format(plane, f)
                    return False
            else:
                if not f in pp.can_fly(plane):
                    print "Error solution invalid, plane {} scheduled to fly invalid flight {}".format(plane, f)
                    return False
    return True

def check_sequencing(pp,s):
    for i, pl in enumerate(s):
        plane = pl[0]
        flights = pl[1:]
        if len(flights) <= 1:
            continue
        for j in range(len(flights) - 1):
            if (flights[j], flights[j+1]) not in pp.can_follow:
                print "Error solution invalid, plane {} scheduled to fly invalid flight sequence ({},{})".format(plane, flights[j], flights[j+1])
                return False
    return True

def check_maintenance(pp,s):
    for i, pl in enumerate(s):
        plane = pl[0]
        flights = pl[1:]
        if len(flights) < pp.min_maintenance_frequency:
            continue
        for j in range(len(flights)-pp.min_maintenance_frequency + 1):
            m_count = 0
            for k in range(pp.min_maintenance_frequency):
                if flights[j+k] in pp.maintenance_flights:
                    m_count += 1
            if m_count == 0:
                print "Error solution invalid, plane {} flys to many flights before maintenance {}".format(plane, flights)
                return False
    return True

def check_all_flights_scheduled_once(pp, s):
    seen = dict()
    for pl in s:
        for f in pl[1:]:
            if f in seen:
                print "Error solution invalid, flight {} scheduled more than once".format(f)
                return False
            seen[f] = True
    for f in pp.flights:
        if not f in seen:
            print "Error solution invalid, flight {} not scheduled".format(f)
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve a plane scheduling csp problem')
    parser.add_argument("p", help="The problem number to solve", type=int)
    parser.add_argument("-a", "--algorithm", help="which backtracking algorithm to use", choices=['BT', 'FC', 'GAC'], default='GAC')
    parser.add_argument("-c", "--allSolns", help="Complete search (Find all solutions)", action="store_true")
    parser.add_argument("-v", "--varHeur", help="Heuristic for selecting next variable to assign", choices=['fixed', 'random', 'mrv'], default='mrv')
    args = parser.parse_args()

    if args.p < 1 or args.p > len(problems):
        print "{} is invalid problem number. I only know about problems {} to {}".format(args.b, 1, len(boards))
        print "If you want to add new problems define them and add them to the list \"problems\""
        exit(1)

    ip = problems[args.p-1]
    print "="*66
    print "Solving problem {}".format(args.p)
    print "Planes: {}".format(ip.planes)
    print "Flights: {}".format(ip.flights)
    print "Solving using {}".format(args.algorithm)
    solns = csp_problems.solve_planes(ip, args.algorithm, args.allSolns, args.varHeur)
    print ""
    for i,s in enumerate(solns):
        print "Solution {}.".format(i+1)
        if not check_plane_solution(ip, s):
            print "ERROR solution is invalid"
        s.sort(key=lambda x: x[0])
        for l in s:
            print "Plane {}: {}".format(l[0], l[1:])
        print "------------------------------\n"
