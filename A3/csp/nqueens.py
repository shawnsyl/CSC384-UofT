import csp_problems
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the n-Queens csp problem')
    parser.add_argument("n", help="the number of queens in the problem", type=int)
    parser.add_argument("-a", "--algorithm", help="which backtracking algorithm to use", choices=['BT', 'FC', 'GAC'], default='BT')
    parser.add_argument("-c", "--allSolns", help="Complete search (Find all solutions)", action="store_true")
    parser.add_argument("-t", "--tablecnstr", help="Use table constraint in csp", action="store_true")
    args = parser.parse_args()

    csp_problems.solve_nQueens(args.n, args.algorithm, args.allSolns, args.tablecnstr)
