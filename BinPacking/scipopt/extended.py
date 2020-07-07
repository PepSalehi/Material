# Marco Chiarandini
# Script not yet verified 

from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
from numpy import round

LOG = True
EPS = 1.e-6  # error margin allowed for rounding


def generate_initial_patterns(s, B):
    # Generate initial patterns with one size for each item width
    t = []
    m = len(s)
    for i in range(m):
        pat = [0]*m  # vector of number of orders to be packed into one bin
        pat[i] = 1
        t.append(pat)
    return t


def solve_pricing_problem(s, B, pi):

    subMIP = Model("Knapsack")     # knapsack sub-problem
    # Turning off presolve
    subMIP.setPresolve(SCIP_PARAMSETTING.OFF)

    # Setting the verbosity level to 0
    subMIP.hideOutput()
    m = len(s)
    subMIP.setMaximize       # maximize
    y = {}
    #print(pi, s)
    for i in range(m):
        y[i] = subMIP.addVar(lb=0, vtype="I", name="y(%s)" % i)

    subMIP.addCons(quicksum(s[i]*y[i] for i in range(m)) <= B, "Width")

    subMIP.setObjective(quicksum(pi[i]*y[i] for i in range(m)), "maximize")

    subMIP.hideOutput()  # silent mode
    subMIP.optimize()
    pat = [round(subMIP.getVal(y[i])) for i in y]
    return subMIP.getObjVal(), pat


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

    K = len(t)

    # master.Params.OutputFlag = 0 # silent mode

    # iter = 0
    while True:
        # print "current patterns:"
        # for ti in t:
        #     print ti
        # print

        # iter += 1
        restricted_master = Model("restricted master LP")  # master LP problem
        restricted_master.setPresolve(SCIP_PARAMSETTING.OFF)
        restricted_master.setHeuristics(SCIP_PARAMSETTING.OFF)
        restricted_master.disablePropagation()
        restricted_master.freeTransform()

        x = {}

        for k in range(K):
            x[k] = restricted_master.addVar(vtype="C", name="x(%s)" %
                                            k)  # note: we consider the LP relaxation

        orders = {}

        for i in range(m):
            orders[i] = restricted_master.addCons(
                quicksum(t[k][i]*x[k] for k in range(K) if t[k][i] > 0) >= 1, "Order(%s)" % i)

        restricted_master.setObjective(quicksum(x[k] for k in range(K)), "minimize")

        restricted_master.optimize()

        pi = [restricted_master.getDualsolLinear(
            c) for c in restricted_master.getConss()]  # keep dual variables

        red_cost, pattern = solve_pricing_problem(s, B, pi)
        if LOG:
            print("objective of knapsack problem:", red_cost, pi)

        if red_cost < 1+EPS:  # break if no more columns
            break

        # newPattern.append(coeff)

        # new pattern
        t.append(pattern)
        if LOG:
            print("shadow prices and new pattern:")
            for (i, d) in enumerate(pi):
                print("\t%5s%12s%7s" % (i, d, pattern[i]))

        # add new column to the master problem
        # Creating new var; must set pricedVar to True
        # restricted_master.freeTransform()
        # newVar = restricted_master.addVar(
        #    "NewPattern_" + str(K+1), vtype="C", obj=1.0, pricedVar=True)

        # Adding the new variable to the constraints of the master problem

        #restricted_master.addConsCoeff(c, newVar, pattern)

        #col = Column()
        # for i in range(m):
        #    if t[K][i] > 0:
        #        col.addTerms(t[K][i], orders[i])
        #x[K] = restricted_master.addVar(obj=1, vtype="C", name="x(%s)" % K, column=col)

        # master.write("MP" + str(iter) + ".lp")
        K += 1

    # restricted_master.optimize()
    return t, x


def solve_MIP_heuristic(s, B, t):
    K = len(t)
    m = len(s)
    restricted_master = Model("restricted master LP")  # master LP problem
    restricted_master.setPresolve(SCIP_PARAMSETTING.OFF)
    restricted_master.setHeuristics(SCIP_PARAMSETTING.OFF)
    restricted_master.disablePropagation()
    restricted_master.freeTransform()

    x = {}

    for k in range(K):
        x[k] = restricted_master.addVar(vtype="I", name="x(%s)" %
                                        k)  # note: we consider the LP relaxation

    orders = {}

    for i in range(m):
        orders[i] = restricted_master.addCons(
            quicksum(t[k][i]*x[k] for k in range(K) if t[k][i] > 0) >= 1, "Order(%s)" % i)

    restricted_master.setObjective(quicksum(x[k] for k in range(K)), "minimize")

    restricted_master.optimize()
    bins = []
    for k in range(K):
        if restricted_master.getVal(x[k]) > 0:
            bins.append([s[i] for i in range(m) if t[k][i] > 0])
    return bins


def solveBinPacking(s, B):
    t, x = solve_master_problem_by_cg(s, B)
    # Finally, solve the IP
    # if LOG:
    #     master.Params.OutputFlag = 1 # verbose mode
    bins = solve_MIP_heuristic(s, B, t)

    # if LOG:
    #     print
    #     print "final solution (integer master problem):  objective =", master.ObjVal
    #     print "patterns:"
    #     for k in x:
    #         if x[k].X > EPS:
    #             print "pattern",k,
    #             print "\tsizes:",
    #             print [w[i] for i in range(m) if t[k][i]>0 for j in range(t[k][i]) ],
    #             print "--> %s rolls" % int(x[k].X+.5)

    # retrieve solution
    return bins


if __name__ == "__main__":
    from data import FFD, Lister, BinPackingExample

    s, B = BinPackingExample()
    s, B = Lister()
    ffd = FFD(s, B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\nBin Packing, column generation:")
    bins = solveBinPacking(s, B)
    print(len(bins), "bins:")
    print(bins)
