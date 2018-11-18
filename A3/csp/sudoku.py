import csp_problems
import backtracking
import argparse

#Sample boards
#List of lists, each internal list is a row. 0 is empty cell, 1-9 is cell fixed to this value
#You can add boards to if you want to solve other boards

b1 = [[0,0,2,0,9,0,0,6,0],
     [0,4,0,0,0,1,0,0,8],
     [0,7,0,4,2,0,0,0,3],
     [5,0,0,0,0,0,3,0,0],
     [0,0,1,0,6,0,5,0,0],
     [0,0,3,0,0,0,0,0,6],
     [1,0,0,0,5,7,0,4,0],
     [6,0,0,9,0,0,0,2,0],
     [0,2,0,0,8,0,1,0,0]]

b2 = [[1,0,6,0,8,0,3,0,0],
      [0,9,7,4,0,1,0,0,0],
      [0,5,0,3,0,0,7,0,0],
      [4,0,0,0,0,7,0,6,0],
      [2,0,0,0,0,0,0,0,8],
      [0,7,0,5,0,0,0,0,9],
      [0,0,3,0,0,9,0,1,0],
      [0,0,0,2,0,3,8,5,0],
      [0,0,8,0,6,0,9,0,4]]

b3 = [[0,7,0,8,5,0,0,0,0],
      [0,9,0,0,0,1,5,0,6],
      [0,0,0,3,0,0,4,0,0],
      [0,3,0,0,0,0,0,0,8],
      [1,0,5,0,0,0,7,0,3],
      [7,0,0,0,0,0,0,2,0],
      [0,0,1,0,0,6,0,0,0],
      [2,0,3,7,0,0,0,6,0],
      [0,0,0,0,3,2,0,1,0]]

b4 = [[0, 0, 0, 0, 0, 6, 0, 0, 0],
      [0, 0, 0, 4, 0, 0, 2, 0, 8],
      [6, 3, 7, 0, 0, 8, 0, 0, 0],
      [2, 4, 0, 0, 0, 0, 0, 9, 0],
      [0, 0, 0, 9, 1, 7, 0, 0, 0],
      [0, 7, 0, 0, 0, 0, 0, 1, 3],
      [0, 0, 0, 3, 0, 0, 6, 8, 1],
      [1, 0, 4, 0, 0, 9, 0, 0, 0],
      [0, 0, 0, 8, 0, 0, 0, 0, 0]]

b5 = [[0, 6, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 7, 0, 0, 0, 0, 0, 4],
      [0, 9, 3, 0, 7, 0, 0, 0, 2],
      [0, 0, 1, 6, 0, 0, 0, 0, 5],
      [0, 0, 0, 8, 4, 2, 0, 0, 0],
      [3, 0, 0, 0, 0, 7, 8, 0, 0],
      [6, 0, 0, 0, 9, 0, 3, 1, 0],
      [7, 0, 0, 0, 0, 0, 5, 0, 0],
      [0, 0, 0, 0, 0, 5, 0, 9, 0]]

b6 = [[7, 0, 0, 1, 6, 0, 0, 0, 0],
      [3, 0, 0, 9, 0, 0, 0, 6, 0],
      [0, 0, 0, 8, 0, 0, 9, 2, 0],
      [0, 0, 6, 0, 1, 0, 0, 5, 0],
      [9, 0, 0, 0, 0, 0, 0, 0, 6],
      [0, 2, 0, 0, 3, 0, 7, 0, 0],
      [0, 1, 3, 0, 0, 2, 0, 0, 0],
      [0, 6, 0, 0, 0, 4, 0, 0, 8],
      [0, 0, 0, 0, 9, 1, 0, 0, 5]]

b7 = [[0, 9, 4, 3, 0, 0, 0, 0, 0],
      [0, 0, 7, 5, 0, 0, 0, 0, 0],
      [0, 0, 1, 4, 0, 0, 0, 2, 0],
      [4, 6, 0, 8, 0, 0, 0, 0, 3],
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [2, 0, 0, 0, 0, 3, 0, 6, 9],
      [0, 5, 0, 0, 0, 6, 2, 0, 0],
      [0, 0, 0, 0, 0, 5, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 6, 4, 0]]

boards = [b1, b2, b3, b4, b5, b6, b7]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve a Sudoku csp problem')
    parser.add_argument("b", help="The board number to solve", type=int)
    parser.add_argument("-a", "--algorithm", help="which backtracking algorithm to use", choices=['BT', 'FC', 'GAC'], default='FC')
    parser.add_argument("-e", "--gacEnforce", help="Don't use search only apply gacEnforce", action="store_true")
    parser.add_argument("-m", "--model", help="Choose CSP model/binary not equals or alldiff", choices=['neq', 'alldiff'], default='neq')
    parser.add_argument("-c", "--allSolns", help="Complete search (Find all solutions)", action="store_true")
    parser.add_argument("-v", "--varHeur", help="Heuristic for selecting next variable to assign", choices=['fixed', 'random', 'mv'], default='fixed')
    args = parser.parse_args()

    if args.b < 1 or args.b > len(boards):
        print "{} is invalid board number. I only know about boards {} to {}".format(args.b, 1, len(boards))
        print "If you want to add new boards add them to the list \"boards\""
        exit(1)

    ib = boards[args.b-1]
    print "="*66
    print "Solving board {}".format(args.b)
    for row in ib:
        print row
    print "-"*60

    if args.gacEnforce:
        print "Applying GAC enforce only using {} model".format(args.model)
        csp = csp_problems.sudokuCSP(ib, args.model)
        backtracking.GacEnforce(csp.constraints(), csp, None, None) #GAC at the root
        vars=csp.variables()
        vars.sort(key=lambda var: var.name())

        print "[",
        for v in vars:
            print "set({}),".format(v.curDomain()),
        print "]"


        var_array = [vars[i:i+9] for i in range(0, len(vars), 9)]
        for row in var_array:
            print "[",
            for v in row:
                print v.curDomain(),
            print "]"
    else:
        print "Solving using {}".format(args.algorithm)
        csp_problems.solve_sudoku(ib, args.model, args.algorithm, args.allSolns, args.varHeur)
