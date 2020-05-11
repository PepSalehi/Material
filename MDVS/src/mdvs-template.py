#!/usr/bin/python3
import pyomo.environ as po
import sys


def readData(F):
    global n, m, k
    data = open(F, 'r')
    header = data.readline().split('\t')
    m = int(header[0])  # Number of depots
    n = int(header[1])  # Number of trips
    k = list(map(lambda z: int(z), header[2:]))
    Ts = []
    for i, line in enumerate(data):
        row = map(lambda z: int(z), line.strip().split('\t'))
        for j, c in enumerate(row):
            if c != -1:
                if i < m and j >= m:  # Pull-out trip
                    Ts += [[i, j, i, c]]
                if i >= m and j < m:  # Pull-in trip
                    Ts += [[(n+m)+i, (n+m)+j, j, c]]
                if i >= m and j >= m:  # Cost of performing j after i
                    for h in range(m):
                        Ts += [[(n+m)+i, j, h, c]]
    # Add reverse arcs
    for i in [i+m for i in range(n)]:
        for h in range(m):
            Ts += [[i, (n+m)+i, h, 0]]
    # Add circulation arcs
    for h in range(m):
        Ts += [[(n+m)+h, h, h, 0]]
    return Ts



# The script starts here
n = 0
m = 0
k = []
# Arcs = tuplelist(readData(sys.argv[1]))
Arcs = readData(sys.argv[1])

# Determine the set of trips S and of depots D
#S = set([i+m for i in range(n)])
S = set([i+m+n+m for i in range(n)])
N = set(range(2*(n+m)))
D = set(range(len(k)))

######################################
# TODO: Change here the capacity for ex 2
######################################
# k = [25,25,25,25]

print("Number of trips:", len(S), " Number of depots:", len(D))

# Model
model = Model("VehicleScheduling")
model.setPresolve(SCIP_PARAMSETTING.OFF)
model.setHeuristics(SCIP_PARAMSETTING.OFF)
model.disablePropagation()
model.setCharParam("lp/initalgorithm", "p")  # let's use the primal simplex
# solving stops, if the relative gap = |primal - dual|/MIN(|dual|,|primal|) is below the given value
model.setParam("limits/gap", 1.0)
# maximal memory usage in MB; reported memory usage is lower than real memory usage! default: 8796093022208
model.setParam("limits/memory", 32000)
model.setParam("limits/time", 10)  # maximal time in seconds to run


# Introduce the arc variables
x = {}
for i, j, h, cost in Arcs:
    # i-j == n+m:  # Circulation arcs (arcs from t_h to s_h)
    if i >= n+m and i < n+m+m and j >= 0 and j < m:
        x[i, j, h] = model.addVar(lb=0.0, ub=k[h], vtype="C", obj=cost,
                                  name="x_"+str(i)+"_"+str(j)+"_"+str(h))
    elif i >= 0 and i < m and j >= m and j < m+n:  # Pull-out trip
        x[i, j, h] = model.addVar(lb=0.0, ub=1.0, vtype="B", obj=cost,
                                  name="x_"+str(i)+"_"+str(j)+"_"+str(h))
    elif i >= m+n+m and i < m+n+m+n and j >= n+m and j < n+m+m:  # Pull-in trip
        x[i, j, h] = model.addVar(lb=0.0, ub=1.0, vtype="B", obj=cost,
                                  name="x_"+str(i)+"_"+str(j)+"_"+str(h))
    elif i >= n+m+m and i < n+m+m+n and j >= m and j < m+n:  # Cost of performing j after i
        x[i, j, h] = model.addVar(lb=0.0, ub=1.0, vtype="B", obj=cost,
                                  name="x_"+str(i)+"_"+str(j)+"_"+str(h))
    elif i >= m and i < m+n and j >= m+n+m and j < m+n+m+n:  # Reverse arcs
        x[i, j, h] = model.addVar(lb=0.0, ub=1.0, vtype="B", obj=cost,
                                  name="x_"+str(i)+"_"+str(j)+"_"+str(h))
    else:
        print("Some arcs not accounted.")

# The objective is to minimize the total costs
model.setMinimize()

# Cover Constraint
for i in S:
    model.addCons(quicksum(x[i, j, h] for s, j, h, c in Arcs if s == i) == 1)

# Flow Balance Constraint
for i in N:
    for h in D:
        BS = [b for b in Arcs if b[1] == i and b[2] == h]  # .select('*',i,h,'*')
        FS = [f for f in Arcs if f[0] == i and f[3] == h]  # Arcs.select(i,'*',h,'*')
        if FS and BS:
            model.addCons(
                quicksum(x[j, i, h] for s, _, _, _ in BS if s == j) ==
                quicksum(x[i, j, h] for _, s, _, _ in FS if s == j))

# Capacity constraint
for h in D:
    model.addCons(x[(n+m)+h, h, h] <= k[h])

model.writeProblem("mdvs.lp")
# Optimize
model.optimize()

if model.getStatus() != "optimal":
    print("Something wrong")
    exit(0)


print('The optimal objective is %g' % model.getObjVal())
# Check number of depots and of vehicles
R = set()
E = set()
#x_star = model.getAttr('X', x)

x_star = {(i, j, h): model.getVal(x[i, j, h]) for i, j, h in x}


###############################################################
# TODO: Change here to get the data you need to fill Table 1
##############################################################
sol = {}
for h in D:
    for s, i, k, c in Arcs:
        if s == h:
            if x_star[h, i, h] > 0.999:
                if h in sol:
                    sol[h] += 1
                else:
                    sol[h] = 1
                R.add(i)
                E.add(h)

print("Vehicles:", len(R), " Depots:", len(E), " Sol:", sol)
