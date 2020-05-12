#!/usr/bin/python3
import argparse
import pyomo.environ as po
import sys


def readData(F):
    data = open(F, 'r')
    header = data.readline().split('\t')
    m = int(header[0])  # Number of depots
    n = int(header[1])  # Number of trips
    k = list(map(lambda z: int(z), header[2:]))
    ######################################
    # TODO: Change here the capacity for ex 2
    ######################################
    # k = [25,25,25,25]
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
    return n, m, k, Ts


def model_mdvs(n, m, k, Arcs):
    # Determine the set of trips S and of depots D
    # S = set([i+m for i in range(n)])
    S = set([i+m+n+m for i in range(n)])
    N = set(range(2*(n+m)))
    D = set(range(len(k)))

    print("Number of trips:", len(S), " Number of depots:", len(D))

    # Model
    model = po.ConcreteModel("VehicleScheduling")

    # Introduce the arc variables
    #model.x = {}
    for i, j, h, cost in Arcs:
        # i-j == n+m:  # Circulation arcs (arcs from t_h to s_h)
        if i >= n+m and i < n+m+m and j >= 0 and j < m:
            model.x[i, j, h] = po.Var(bounds=(0, k[h]), within=po.Reals)
        elif i >= 0 and i < m and j >= m and j < m+n:  # Pull-out trip
            model.x[i, j, h] = po.Var(bounds=(0.0, 1.0), within=po.Binary)
        elif i >= m+n+m and i < m+n+m+n and j >= n+m and j < n+m+m:  # Pull-in trip
            model.x[i, j, h] = po.Var(bounds=(0.0, 1.0), within=po.Binary)
        elif i >= n+m+m and i < n+m+m+n and j >= m and j < m+n:  # Cost of performing j after i
            model.x[i, j, h] = po.Var(bounds=(0.0, 1.0), within=po.Binary)
        elif i >= m and i < m+n and j >= m+n+m and j < m+n+m+n:  # Reverse arcs
            model.x[i, j, h] = po.Var(bounds=(0.0, 1.0), within=po.Binary)
        else:
            print("Some arcs not accounted.")

    # The objective is to minimize the total costs
    model.obj = po.Objective(expr=sum(model.x[i, j, h]*cost for i, j, h, cost in Arcs))

    # Cover Constraint
    model.cover = po.ConstraintList()
    for i in S:
        model.cover.add(expr=sum(model.x[i, j, h] for s, j, h, c in Arcs if s == i) == 1)

    # Flow Balance Constraint
    model.flow_balance = po.ConstraintList()
    for i in N:
        for h in D:
            BS = [b for b in Arcs if b[1] == i and b[2] == h]  # .select('*',i,h,'*')
            FS = [f for f in Arcs if f[0] == i and f[3] == h]  # Arcs.select(i,'*',h,'*')
            if FS and BS:
                model.flow_balance.add(
                    expr=sum(model.x[j, s, h] for j, s, k, _ in BS if s == i and k == h) - sum(model.x[s, j, h] for s, j, k, _ in FS if s == i and h == k) == 0)

    # Capacity constraint
    model.capacity_cons = po.ConstraintList()
    for h in D:
        model.capacity_cons.add(expr=model.x[(n+m)+h, h, h] <= k[h])

    model.pprint()
    # model.write("mdvs.lp")
    # Optimize
    results = po.SolverFactory("glpk").solve(m, tee=True)

    if str(results.Solver.status) != 'ok':
        print("Something wrong")
        exit(0)

    print('The optimal objective is %g' % model.obj())
    # Check number of depots and of vehicles
    R = set()
    E = set()
    # x_star = model.getAttr('X', x)

    x_star = {(i, j, h): model.x[i, j, h]() for i, j, h in x}

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


def main():
    parser = argparse.ArgumentParser(description='MILP solver for timetabling.')
    parser.add_argument(dest="filename", type=str, help='filename')
    parser.add_argument("-e", "--example", type=str, dest="example",
                        default="value", metavar="[value1|value2]", help="Explanation [default: %default]")

    args = parser.parse_args()  # by default it uses sys.argv[1:]

    n, m, k, Arcs = readData(args.filename)
    model_mdvs(n, m, k, Arcs)


if __name__ == "__main__":
    main()
