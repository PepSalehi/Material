import pyomo.environ as po
from data import BinPackingExample, Lister, FFD


def bpp(s, B):
    n = len(s)
    U = len(FFD(s, B))
    model = po.ConcreteModel("bpp")
    x, y = {}, {}
    for i in range(n):
        for j in range(U):
            x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))
    for j in range(U):
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)

    for i in range(n):
        model.addCons(quicksum(x[i, j] for j in range(U)) == 1, "Assign(%s)" % i)
    for j in range(U):
        model.addCons(quicksum(s[i]*x[i, j] for i in range(n)) <= B*y[j], "Capac(%s)" % j)
    for j in range(U):
        for i in range(n):
            model.addCons(x[i, j] <= y[j], "Strong(%s,%s)" % (i, j))
    model.setObjective(quicksum(y[j] for j in range(U)), "minimize")
    model.data = x, y
    return model


def solveBinPacking(s, B):
    n = len(s)
    U = len(FFD(s, B))
    model = bpp(s, B)
    x, y = model.data
    model.optimize()
    bins = [[] for i in range(U)]
    for (i, j) in x:
        if model.getVal(x[i, j]) > .5:
            bins[j].append(s[i])
    for i in range(bins.count([])):
        bins.remove([])
    for b in bins:
        b.sort()
    bins.sort()
    return bins


if __name__ == '__main__':
    # s, B = BinPackingExample()
    s, B = Lister()
    bins = solveBinPacking(s, B)
    print(len(bins))
    for b in bins:
        print((b, sum(b)))
