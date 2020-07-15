import time
import math

import itertools


def heuristic(data):
    V = list(range(data.cust+2))
    Vprime = V.copy()
    Vprime.remove(0)
    Vprime.remove(9)
    A = [(i, j) for i, j in itertools.product(V, V) if i != j]
    A.remove((0, 9))
    A.remove((9, 0))
    
    routes=[]
    totCost=0
    while len(Vprime)>0: # start a route
        route=[0]
        i=0
        demand = 0
        time = 0
        while len(Vprime)>0: # increase the route
            bestCost=math.inf
            best_j=0
            for j in Vprime:
                if data.cost[i,j] < bestCost:
                    if demand + data.demand[j-1] <= data.Q and time + data.timeCost[i,j] <= data.twEnd[j]:
                        bestCost=data.cost[i,j]
                        best_j=j
            if best_j==0: # close route
                break
            else: # continue the route            
                totCost += data.cost[i,j]
                demand += data.demand[j-1]
                time = max(data.twStart[j], time + data.timeCost[i,j])
                route+=[j]
                i=j
                Vprime.remove(j)
        totCost += data.cost[route[-1],9]
        route += [0]
        routes+=[route]
    return(totCost,routes)


if __name__ == "__main__":
    import sys 
    sys.path.append("data/")
    from data import *
    cost, routes = heuristic(data)
    print(cost, routes)
    data.plot_routes(routes)
