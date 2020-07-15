import numpy as np
import matplotlib
import itertools
import math
import matplotlib.pyplot as plt
from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
import sys
import time
import functools
# ==============================================================================
# ==============================================================================


class PrunedByInfeasibility(Exception):
    """Exception raised for infeasible RMP
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class PrunedByBounding(Exception):
    """Exception raised for infeasible RMP
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


# ==============================================================================
# ==============================================================================
def solveReducedMaster(data, routes, costR, routes_arcs, subset_rows, iteration=-1, arcs_to_zero=[], arcs_to_one=[]):
    print("\n==============================================")
    print("Solving the reduced master problem... with {} columns {}".format(len(routes),len(routes_arcs)))

    assert(len(routes)==len(routes_arcs))

    # model
    m = Model("ReducedMaster")
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.disablePropagation()
    m.freeTransform()
    m.hideOutput()

    Route_indices = list(range(len(routes)))
    # zero branching constraints
    for a in arcs_to_zero:
        for r in Route_indices: #range(len(routes_arcs)):
            if a in routes_arcs[r]:
                Route_indices.remove(r)

    # one branching constraints
    for a in arcs_to_one:
        for r in Route_indices: #range(len(routes_arcs)):
            for x in [(a[0],j) for j in data.Vprime+[9] if j not in a and (a[0],j) != (0,9)]+[(i,a[1]) for i in data.Vprime+[0] if i not in a and (i,a[1])!=(0,9)]:
                if x in routes_arcs[r]:
                    Route_indices.remove(r)
                    break
    #print(routes[Route_indices,])

    print("reduced to: {}".format(len(Route_indices)))
    # variables
    theta = {}
    for i in Route_indices:
        theta[i] = m.addVar(lb=0.0, vtype="C", name="theta" + str(i))

    # constraints
    m.addCons(quicksum(theta[r] for r in Route_indices) <= data.m, name="Vehicles")

    for i in data.Vprime:
        m.addCons(quicksum(routes[r][i]*theta[r]
                           for r in Route_indices) >= 1, name="Costumer" + str(i))

    for (i, j, k) in subset_rows:
        m.addCons(quicksum(b(i, j, k, routes[r])*theta[r]
                           for r in Route_indices) <= 1, name="Cut" + str(i) + str(j) + str(k))

    # objective
    m.setObjective(quicksum(costR[r]*theta[r] for r in Route_indices), "minimize")

    # optimize
    m.writeProblem("master_"+str(iteration)+".lp")
    m.optimize()

    if m.getStatus() == "optimal":
        opt_val = m.getObjVal()
        print('Optimal solution for master problem is ' + str(opt_val))

        #for v in m.getVars():
        #    if m.getVal(v) > 0.0:
        #        print(v.name, " = ", m.getVal(v))

        solution = np.zeros(len(routes))
        solution[Route_indices] = np.array([m.getVal(v) for v in m.getVars()])

        duals = np.array([m.getDualsolLinear(c) for c in m.getConss()])
        # duals = [m.getDualMultiplier(c) for c in m.getConss()]
        print("Theta vars: ",np.around(solution,3))
        print("Dual values: ", np.around(duals,3))
        # print("Reduced costs: ",[m.getVarRedcost(v) for v in m.getVars()])
    elif m.getStatus()=="infeasible":
        raise PrunedByInfeasibility("RMP infeasible")
    else:
        print("Something is wrong in the master problem. Status ",m.getStatus())

    print("==============================================")
    return (duals, solution, opt_val)
# ==============================================================================
# ==============================================================================


# ==============================================================================
# ==============================================================================
def SolvePricing(dual, data, subset_rows, iteration=-1, arcs_to_zero=[], arcs_to_one=[]):
    print("==============================================")
    print("Solving the pricing problem... ")

    # ==========================================================================
    # build pricing problem
    p = Model("PricingProblem")
    p.setPresolve(SCIP_PARAMSETTING.OFF)
    p.setHeuristics(SCIP_PARAMSETTING.OFF)
    p.disablePropagation()
    p.hideOutput()
    # ==========================================================================

    # ==========================================================================
    # variables
    x = {}
    for (i, j) in data.A:
        x[i, j] = p.addVar(vtype="B", name="x" + str(i) + str(j))

    t = {}
    for i in data.V:
        t[i] = p.addVar(vtype="C", lb=data.twStart[i], ub=data.twEnd[i], name="t" + str(i))

    Z = {}
    for I in subset_rows:
        Z[I] = p.addVar(vtype="B", name="Z" + "_".join(map(str, I)))

    # ==========================================================================

    # ==========================================================================
    # constraints
    # all routes are a path from source to target
    # all routes created by leaving the depot
    p.addCons(quicksum(x[0, j] for j in data.Vprime) == 1, name="from_source_v")
    # mass balance at inner nodes
    for i in data.Vprime:
        p.addCons(quicksum(x[i, j] for j in data.Vprime+[9] if j != i) ==
                  quicksum(x[j, i] for j in data.Vprime+[0] if j != i),
                  name="balance_%d" % (i))
    # all routes terminate at target node
    p.addCons(quicksum(x[i, 9] for i in data.Vprime) == 1, name="target_v")

    # time constraints
    for (i, j) in data.A:
        p.addCons(t[i]+data.timeCost[i, j]-t[j] <= (1-x[i, j])
                  * data.M[i, j], name="time_%d_%d" % (i, j))

    # Capacity
    p.addCons(quicksum(data.demand[i]*x[i, j] for i in data.Vprime for j in data.Vprime +
                       [9] if j != i) <= data.Q, name="Capacity on vehicle")

    # Subset row cuts
    for (i, j, k) in subset_rows:
        p.addCons(quicksum(x[i, s] for s in data.Vprime+[9] if s != i)
                  # + quicksum(x[s, i] for s in Vprime+[0] if s != i)
                  + quicksum(x[j, s] for s in data.Vprime+[9] if s != j)
                  # + quicksum(x[s, j] for s in Vprime+[0] if s != j)
                  + quicksum(x[k, s] for s in data.Vprime+[9] if s != k)
                  # + quicksum(x[s, k] for s in Vprime+[0] if s != k)
                  <= 2*Z[i, j, k] + 1, name="subset_rows")
    # ==========================================================================
    for a in arcs_to_zero:
        p.addCons(x[a]==0, name="zero_branch")
    for a in arcs_to_one:
        p.addCons(quicksum(x[a[0],j] for j in data.Vprime+[9] if j!=a[1] and j!=a[0] and (a[0],j) != (0,9))
        +quicksum(x[i,a[1]] for i in data.Vprime+[0] if i!=a[0] and i!=a[1] and (i,a[1])!=(0,9))==0, name="one_branch")

    # ==========================================================================
    # objective
    p.setObjective(quicksum(data.cost[i, j] * x[i, j] for (i, j) in data.A)
                   - quicksum(1.0*dual[i] * x[i, j]
                              for i in data.Vprime for j in data.Vprime+[9] if j != i)
                   - 1.0*dual[0]
                   - quicksum(dual[data.cust+1+ell]*Z[subset_rows[ell]]
                              for ell in range(len(subset_rows))),
                   "minimize")
    # remember dual[8] is lambda_0
    # ==========================================================================

    # ==========================================================================
    # optimize
    if iteration>=0:
        p.writeProblem("subproblem_"+str(iteration)+".lp")
    p.optimize()

    # print solution
    if p.getStatus() == "optimal":
        z = p.getObjVal()

        print('Optimal solution for pricing problem is ' + str(z))
        #print('\nThe variables are:')

        #for v in p.getVars():
        #    if p.getVal(v) > 1e-08:
        #        print(v.name, " = ", p.getVal(v))
        #for (i, j, k) in Z:
        #    print("Z", (i, j, k), p.getVal(Z[i, j, k]))

        cost = 0
        route = np.zeros(data.cust+2)
        route[0] = 1
        route[data.cust+1] = 1
        for i in data.Vprime:
            for j in data.V:
                if j != i:
                    if p.getVal(x[i, j]) > 1e-08:
                        route[i] = 1

        arcs = []
        for (i, j) in data.A:
            if p.getVal(x[i, j]) > 1e-08:
                arcs += [(i, j)]

        cost = 0
        for (i, j) in data.A:
            if p.getVal(x[i, j]) > 1e-08:
                cost += data.cost[i, j]
        print("Selected arcs: ",arcs, " Nodes: ",list(filter(lambda i: route[i] > 0, data.V)), " Cost: ", cost)
        #price1 = sum([data.cost[i][j] for (i, j) in A if p.getVal(x[(i, j)]) > 0.5])
        #price2 = sum([dual[i-1] for i, b in enumerate(route) if b == 1])+dual[data.cust]
        #print(price1, price2, price1-price2)
        print("==============================================")
        return (z, route, arcs, cost)
    elif p.getStatus()=="infeasible":
        raise Infeasible("pricing problem infeasible")
    else:
        print("Something is wrong in the pricing problem")
        sys.exit(0)




# ==============================================================================


# ==============================================================================
# ==============================================================================
def b(i, j, k, Route):
    if Route[i]+Route[j]+Route[k] >= 2:
        return 1
    else:
        return 0
# ==============================================================================
# ==============================================================================


# ==============================================================================
# ==============================================================================
def find_subset_rows(solution, routes, data):
    print("="*80)
    print("Solving the separation problem")
    R = range(len(routes))
    for (i, j, k) in itertools.combinations(data.Vprime, 3):
        if (sum(b(i, j, k, routes[r])*solution[r] for r in R) > 1):
            return (i, j, k)
    print('No subset row')
    return None


def branching_rule(solution, routes_arcs, arcs_used):
    if True:
        # select arc with most fractional total flow: that is arc maybe the most uncertain to include.  
        arcs = dict()
        for i in range(len(solution)):
            if solution[i] > 0 and solution[i]<1:
                for a in routes_arcs[i]:
                    if a[0]!=0 and a[1]!=9 and a not in arcs_used:
                        if a in arcs:
                            arcs[a]+=solution[i]
                        else:
                            arcs[a]=solution[i]
        arcs = {x: min(arcs[x]-math.floor(arcs[x]),math.ceil(arcs[x])-arcs[x]) for x in arcs}
        choices = list(filter(lambda x: arcs[x]==max(arcs.values()),arcs))
        print(arcs)
        return choices[np.random.randint(len(choices))]
    else:
        # select the arcs in the most fractional variable
        indices = np.where(solution==max( map(min, zip(solution-np.floor(solution),np.ceil(solution)-solution)  )))
        print("Most fractional vars: ",indices[0])
        #for i in indices[0]:
        #    print(routes_arcs[i])
        branching_arcs = functools.reduce(lambda x,y: x+y, [routes_arcs[i] for i in indices[0]] )
        branching_arcs = [x for x in branching_arcs if x[0]!=0 and x[1]!=9 and x not in arcs_used]
        return branching_arcs[np.random.randint(len(branching_arcs))]
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
def solve_extended(data, routes, costR, routes_arcs, SRC=False, BB=False, arcs_to_zero=[], arcs_to_one=[], LB=-math.inf, UB=math.inf):
    # starting value for z
    z = -1
    subset_rows = []
    iteration=0
    
    while True:
        iteration+=1
        print("----------------------ITERATION {}----------------------".format(iteration))

        try:
            (duals, solution, opt_val) = solveReducedMaster(data, routes, costR, routes_arcs, subset_rows, iteration, arcs_to_zero, arcs_to_one)
            #tmp = input("Hit Enter")
        except PrunedByInfeasibility as inf:
            print(inf.message)
            raise PrunedByInfeasibility(inf.message)

        try:
            (z, route, arcs, cost) = SolvePricing(duals, data, subset_rows, iteration, arcs_to_zero, arcs_to_one)
            #tmp = input("Hit Enter")
        except PrunedByInfeasibility as inf:
            print(inf.message)
            break

        modified = False
        if z < -1e-08:  # we have negative reduced costs, add new columns
            modified = True
            # add new row
            routes = np.concatenate((routes, [route]), axis=0)
            # add new cost
            costR = np.concatenate((costR, [cost]), axis=0)

            routes_arcs += [arcs]
            print("Add new column to RMP: ", route)
            print("Costs:", costR)
        elif SRC:  # Cuts:
            rows = find_subset_rows(solution, routes, data)
            if rows is not None:
                modified = True
                subset_rows.append(rows)
                print('Add Subset Row Cut: ', rows)
        if not modified:
            break
        #tmp = input("Hit Enter")

    #print("==============================================")
    print("CG process finished")
    print('Optimal solution value for RMP is: ' + str(opt_val))
    print("Theta vars: ",solution)
    if any(solution-np.floor(solution))<0.001: # sol integer
        print("solution INTEGER")
        #print("Routes with value >0")
        tours=[]
        for i in range(len(solution)):
            if solution[i] > 0:
                tour = list(filter(lambda j: routes[i][j] > 0, data.V))
                tours += [tour[:-1]+[0]]
                #print(tour)
        #print(tours)
        UB=min(UB,opt_val)
        if abs(UB-LB)<0.0001:
            print("OPTIMAL SOLUTION FOUND:")
            print(opt_val, tours)
            raise SystemExit
            return opt_val, tours
        return opt_val, tours
    else:
        print("solution NOT INTEGER")
        if BB:
            if LB==-math.inf:
                LB=opt_val
            if opt_val>UB:
                raise PrunedByBounding("Pruned by bounding: LB= {}, UB= {} ".format(LB,UB))
            branching_arc = branching_rule(solution, routes_arcs, set(arcs_to_zero) | set(arcs_to_one))
            print("Branching arc: ", branching_arc)
            tmp = input("Left branch: ")
            branches=[]
            try:
                print("+++++++++++++++++++++++++++++  OPEN NEW NODE LB={} UB={} ++++++++++++++++++++++++++++++++++++++++++".format(LB,UB))
                print("zero arcs: ", arcs_to_zero+[branching_arc])
                print("one arcs: ", arcs_to_one)
                res_l = solve_extended(data,  routes, costR, list(routes_arcs), SRC, BB, arcs_to_zero+[branching_arc], arcs_to_one, LB, UB) # left branch
                branches += [res_l]
                LB=min(LB,branches[0][0])
            except PrunedByInfeasibility as inf:
                print(inf.message)
            except PrunedByBounding as prn:
                print(prn.message)
            tmp = input("Right branch: ")
            try:
                print("+++++++++++++++++++++++++++++  OPEN NEW NODE LB={} UB={} ++++++++++++++++++++++++++++++++++++++++++".format(LB,UB))
                print("zero arcs: ", arcs_to_zero)
                print("one arcs: ", arcs_to_one+[branching_arc])
                res_r = solve_extended(data, routes, costR, list(routes_arcs), SRC, BB, arcs_to_zero, arcs_to_one+[branching_arc], LB, UB) # right branch
                branches += [res_r]
            except PrunedByInfeasibility as inf:
                print(inf.message)
            except PrunedByBounding as prn:
                print(prn.message)
            if len(branches)==0:
                raise PrunedByInfeasibility("both branches infeasible")
            i = list(filter(lambda k: branches[k][0]==min([x[0] for x in branches]), range(len(branches))))
            LB=min(LB,min([x[0] for x in branches]))
            return branches[i[0]][0], branches[i[0]][1]
        else:
            return opt_val, []
    # ==============================================================================




def enrich_data(data):
    # We enrich the data class
    # All vertices including depots
    data.V = range(data.cust+2)
    data.depots = [0, 9]
    # All vertices except depot
    data.Vprime = [i for i in data.V if i not in data.depots]
    # All arcs
    data.A = [(i, j) for i, j in itertools.product(data.V, data.V) if i != j]
    data.A.remove((0, 9))
    data.A.remove((9, 0))

    data.M = {(i, j): max(0, data.twEnd[i]+data.timeCost[i, j]-data.twStart[j]) for (i, j) in data.A}

    I = np.concatenate((np.ones((len(data.Vprime),1),dtype=int),np.identity(len(data.Vprime),dtype=int),np.ones((len(data.Vprime),1),dtype=int)),axis=1)
    data.routes = np.concatenate((data.routes, I), axis=0)
    #data.costRoutes += np.array([data.cost[0,v]+data.cost[v,9] for v in data.Vprime])
    data.costRoutes = np.concatenate((data.costRoutes,np.array([data.cost[0,v]+data.cost[v,9] for v in data.Vprime])))

    data.routes_arcs=[]
    for ell in range(len(data.routes)):
        vs = np.array(list(filter(lambda i: data.routes[ell][i]>0, data.V)))
        while True:
            cost = sum([data.cost[vs[i],vs[i+1]] for i in range(vs.size-1)])
            if cost == data.costRoutes[ell]:
                break
            np.random.shuffle(vs[1:-1])
        data.routes_arcs += [[(vs[i],vs[i+1]) for i in range(vs.size-1)]]


if __name__=="__main__":

    SRC = False # with cuts
    BB = True

    sys.path.append("data")
    from data.data import data

    np.random.seed(3)
    enrich_data(data)
    cost, routes = solve_extended(data, data.routes, data.costRoutes, data.routes_arcs, SRC, BB)
    print("Solution: ")
    print(cost, routes)
    data.plot_routes(routes)
