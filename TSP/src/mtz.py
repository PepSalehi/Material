from tsputil import *
from pyscipopt import *
from collections import OrderedDict


##################################################
MY_ID = 25  # YOUR_ID
##################################################


def solve_tsplp(points, subtours=[]):
    points = list(points)
    V = range(len(points))
    E = [(i, j) for i in V for j in V if i != j]

    m = Model("TSP0")
    # m.setPresolve(SCIP_PARAMSETTING.OFF)
    # m.setHeuristics(SCIP_PARAMSETTING.OFF)
    # m.disablePropagation()
    # m.setCharParam("lp/initalgorithm", "p")  # let's use the primal simplex
    # solving stops, if the relative gap = |primal - dual|/MIN(|dual|,|primal|) is below the given value
    #m.setParam("limits/gap", 1.0)
    # maximal memory usage in MB; reported memory usage is lower than real memory usage! default: 8796093022208
    m.setParam("limits/memory", 32000)
    m.setParam("limits/time", 100)  # maximal time in seconds to run

    # BEGIN: Write here your model for Task 2
    x = OrderedDict()
    for (i, j) in E:
        x[i, j] = m.addVar(lb=0.0, ub=1.0, obj=distance(points[i], points[j]),
                           vtype="B", name="x[%d,%d]" % (i, j))
    u = {}
    for i in V[1:]:
        u[i] = m.addVar(lb=0.0, ub=len(V)-2, obj=0,
                        vtype="C", name="u[%d]" % (i))

    # Objective
    m.setMinimize()

    # Constraints
    for v in V:
        m.addCons(quicksum(x[v, i] for i in V if (v, i) in E) == 1, "out_balance_%d" % v)
        m.addCons(quicksum(x[i, v] for i in V if (i, v) in E) == 1, "in_balance_%d" % v)

    for (i, j) in E:
        if i == 0 or j == 0:
            continue
        m.addCons(u[i]+1 <= u[j] + len(V)*(1-x[i, j]), "u_%d_%d" % (i, j))
    # END

    m.writeProblem("tsplp.lp")
    m.optimize()

    if m.getStatus() == "optimal":
        print('The optimal objective is %g' % m.getObjVal())
        m.writeSol(m.getSols()[0], "tsplp.sol")  # write the solution
        return {(i, j): m.getVal(x[i, j]) for i, j in x}
    else:
        print("Something wrong in solve_tsplp")
        exit(0)


import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # BEGIN: Update this part with what you need
        points = Cities(20, seed=MY_ID)
        # plot_situation(points)
        lpsol = solve_tsplp(points)
        plot_situation(points, lpsol)
        # cutting_plane_alg(points)
        # END
    elif len(sys.argv) == 2:
        # BEGIN: Update this part for Task 7 and on
        locations = read_instance(sys.argv[1])
        # plot_situation(locations)
        # lpsol = solve_tsplp(locations)
        # plot_situation(locations, lpsol)
        # cutting_plane_alg(locations)
        # END
    else:
        print('Use either with no input file or with an input file argument.')
