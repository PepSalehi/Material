from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
from numpy import round
from data import FFD, BinPackingExample

LOG = True
EPS = 1.e-6  # error margin allowed for rounding


def generate_initial_patterns(s, B):
    # Generate initial patterns with one size for each item width
    t = []
    m = len(s)
    for i in range(m):
        pat = [0]*m  # vector of number of orders to be packed into one bin
        pat[i] = int(B/s[i])
        t.append(pat)
    return t


def solve_pricing_problem(s, B, pi):

    subMIP = po.ConcreteModel("Knapsack")     # knapsack sub-problem
    # Turning off presolve
    subMIP.setPresolve(SCIP_PARAMSETTING.OFF)

    # write here the model

    # solve
    solver = po.SolverFactory('glpk')
    solver.options['tmlim'] = 10  # seconds
    solver.options['noscale'] = True  # seconds
    results = solver.solve(subMIP)
    # extract the pattern found
    pattern = []

    return subMIP.getObjVal(), pattern


def solve_master_problem_by_cg(s, B):
    """ use column generation (Gilmore-Gomory approach).
    Parameters:
        - s: list of item's width
        - B: bin/roll capacity
    Returns a solution: list of lists, each of which with elements in the bin.
    """
    m = len(s)
    t = generate_initial_patterns(s, B)

    if LOG:
        print("sizes of orders=", s)
        print("bins size=", B)
        print("initial patterns", t)

    iter = 0
    while True:
        # print "current patterns:"
        # for ti in t:
        #     print ti
        # print
        K = len(t)
        restricted_master = po.ConcreteModel("restricted master LP")  # master LP problem
        # restricted_master.setPresolve(SCIP_PARAMSETTING.OFF)
        # restricted_master.setHeuristics(SCIP_PARAMSETTING.OFF)
        # restricted_master.disablePropagation()
        # restricted_master.freeTransform()

        ####
        # YOUR CODE
        ####

        iter += 1
        restricted_master.optimize()
        pi = [restricted_master.getDualsolLinear(
            c) for c in restricted_master.getConss()]  # keep dual variables

        red_cost, pattern = solve_pricing_problem(s, B, pi)
        if LOG:
            print("objective of knapsack problem:", red_cost)

        if red_cost < 1+EPS:  # break if no more columns
            break

        # newPattern.append(coeff)

        # new pattern
        t.append(pattern)
        # if LOG:
        #     print "shadow prices and new pattern:"
        #     for (i,d) in enumerate(pi):
        #         print "\t%5s%12s%7s" % (i,d,pat[i])
        #     print

        # add new column to the master problem
        # Creating new var; must set pricedVar to True

    return t, x


def solveBinPacking(s, B):
    t, x = solve_master_problem_by_cg(s, B)
    # Finally, solve the IP
    # if LOG:
    #     master.Params.OutputFlag = 1 # verbose mode
    x = solve_MIP_heuristic.optimize(s, B, t)

    # retrieve solution
    bins = []

    return bins


if __name__ == "__main__":
    s, B = BinPackingExample()
    ffd = FFD(s, B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\nBin Packing, column generation:")
    bins = solveBinPacking(s, B)
    print(len(bins), "rolls:")
    print(bins)
