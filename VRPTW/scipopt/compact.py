from numpy import *
import itertools
from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum

def SolveCompactModel(data):
    print("==============================================")
    print("Solving the compact model... \n")

    # ==========================================================================
    # build problem
    p = Model("Compact Model")
    # p.setPresolve(SCIP_PARAMSETTING.OFF)
    # p.setHeuristics(SCIP_PARAMSETTING.OFF)
    # p.disablePropagation()
    # p.hideOutput()
    p.redirectOutput()
    # ==========================================================================

    # ==========================================================================
    # All vertices
    V = range(data.cust+2)
    depot = [0, 9]
    # All vertices except depot
    Vprime = [i for i in V if i not in depot]
    # All arcs
    A = [(i, j) for i, j in itertools.product(V, V) if i != j]
    A.remove((0, 9))
    A.remove((9, 0))
    # ==========================================================================
    Vehicles = range(3)
    M = {(i,j): max(0,data.twEnd[i]+data.timeCost[i,j]-data.twStart[j]) for (i,j) in A}
    # ==========================================================================

    # variables
    x = {}
    for (i, j) in A:
        for k in Vehicles:
            x[i, j, k] = p.addVar(vtype="B", name="x[%d,%d,%d]" % (i, j, k))

    t = {}
    for k in Vehicles:
        for i in V:
            t[i,k] = p.addVar(vtype="C", lb=data.twStart[i], ub=data.twEnd[i], name="t[%d,%d]" % (i,k))

    # ==========================================================================
    # ==========================================================================
    # all vertices visited by exiting
    for i in Vprime:
        p.addCons(quicksum(x[i, j, k] for j in Vprime+[9] if j!= i for k in Vehicles) == 1, name="visit_all_%d" % i)

    # all routes are a path from source to target
    # all routes created by leaving the depot
    for k in Vehicles:
        p.addCons(quicksum(x[0, j, k] for j in Vprime) == 1, name="from_source_v_%d" % (k))
    # mass balance at inner nodes
    for k in Vehicles:
        for i in Vprime:
            p.addCons(quicksum(x[i, j, k] for j in Vprime+[9] if j!=i) ==
                      quicksum(x[j, i, k] for j in Vprime+[0] if j!=i), name="balance_%d_%d" % (i,k))
    # all routes terminate at target node
    for k in Vehicles:
        p.addCons(quicksum(x[i, 9, k] for i in Vprime) == 1, name="target_v_%d" % k)


    # time constraints
    for k in Vehicles:
        for (i,j) in A:
            p.addCons(t[i,k]+data.timeCost[i,j]-t[j,k]<=(1-x[i,j,k])*M[i,j],name="time %d%d%d" % (i,j,k) )

    # Capacity
    for k in Vehicles:
        p.addCons(quicksum(data.demand[i-1]*x[i,j,k] for i in Vprime for j in Vprime+[9] if j!=i ) <= data.Q, name="Capacity on vehicle %d" % k)


    # ==========================================================================
    # objective
    p.setObjective(quicksum(data.cost[i,j] * x[i, j, k] for (i, j) in A for k in Vehicles), "minimize")
    # ==========================================================================

    # ==========================================================================
    # optimize
    #p.writeProblem("compact_model.lp")
    p.optimize()

    # print solution
    if p.getStatus() == "optimal":
        z = p.getObjVal()

        print('Optimal solution ' + str(z))
        print('\nThe variables are:')

        for v in p.getVars():
            if p.getVal(v) > 1e-08:
                print(v.name, " = ", p.getVal(v))

        cost = 0
        routes=[]
        route = zeros(data.cust)
        for k in Vehicles:
            arcs=[]
            for (i,j) in A:
                if p.getVal(x[i, j, k]) > 1e-08:
                    if j==9:
                        j=0
                    arcs+=[(i,j)]
                    #route[i-1] = 1
            routes+=[arcs]
        cost = 0
        for k in Vehicles:
            #print("Route: ",k)
            for (i, j) in A:
                if p.getVal(x[i, j, k]) > 1e-08:
                    #print(i,j)
                    cost += data.cost[i, j]
        return cost, routes
    else:
        print("Something is wrong in the fomulation")
        sys.exit(0)



if __name__ == "__main__":
    import sys 
    sys.path.append("data/")
    from data.data import data
    cost, arcs = SolveCompactModel(data)
    print(cost, arcs)
    data.plot_routes_arcs(arcs)