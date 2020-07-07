import sys

from gurobipy import *
from subprocess import *

tmpdir=""

# Data format:
# <n> number of vertices <m> number of arcs
# <i> <j> <c>  arc (i,j) with cost c
def readDataNoResource(F):
    data = open(F, 'r')
    n,m,k,s,t = data.readline().split()
    U = map(lambda z: int(z), data.readline().split())
    A = []
    for line in data:
        arc = line.strip().split()
        if int(arc[0]) <= t and int(arc[1]) <= t:
            A += [(int(arc[0]),int(arc[1]),float(arc[2]))]
    return set(range(int(t))), A, int(s), int(t)

# Read data and return a list of arcs with resource consumption:
# <i> <j> <c> <r_1> ... <r_k> : from i to j with cost c and resource r_i
def readDataWithResource(F):
    data = open(F, 'r')
    n,m,k,s,t = data.readline().split()
    U = map(lambda z: int(z), data.readline().split())
    A = []
    for line in data:
        arc = line.strip().split()
    A += [(int(arc[0]),int(arc[1]),float(arc[2]),map(lambda z: int(z), data.read().split()))]
    return set(range(int(n))), A, k, s, t


# Solve a shortest path problem from 's' to 't' in
# the graph G=(N,A)
def pricingProblem(N,A,s,t):
    model = Model("ShortestPath")
    model.setParam(GRB.param.OutputFlag, 0)
    ########### TODO ###################################
    # Define Flow Node Balance
     # Add arc flow variables

    # The objective is to minimize the total costs
    model.modelSense = GRB.MINIMIZE
    # Update model to integrate new variables
    model.update()
    #  Flow Balance Constraint

     ########### END TODO ###################################
    # Optimize
    model.optimize()
    # Check Solver Status
    if model.status != GRB.status.OPTIMAL:
        print 'Shortest Path - ERROR:',model.status
        exit(0)
    # Recover the solution
    n_sol = model.getAttr('SolCount')
    Ps = {}
    for k in range(n_sol):
        model.setParam(GRB.Param.SolutionNumber, k)
        obj = 0.0
        Path = []
        for i,j in x:
            if x[i,j].getAttr('Xn') > 0.99:
                obj += x[i,j].getAttr('Obj')
                Path += [(i,j)]
        Ps[k] = (obj, Path)
    return Ps

def ResourceConstrainedShortestPath(N,A,s,t):
    global tmpdir
    paths={}
    try:
        childid = os.fork()
    except OSError:
        print "Error creating child ".format(OSError)
    print childid
    if childid==0: # we are the children
        filename=tmpdir+"/"+str(os.getpid())+".txt"
        f = open(filename, 'w')
        f.write("%d %d %d %d %d\n" % (len(N),len(B),s,t,1))
        f.write("2\n")
        for i,j,c in B:
            f.write("%d %d %d %d\n" % (i,j,c,1))
        f.close()
        try:
            stdout_paths = check_output(["path-solver",filename])
        except CalledProcessError as e:
            print "failed {0}\n returned {1}".format(e.cmd, e.returncode)
        print stdout_paths.strip()
        rows = stdout_paths.strip().split("\n")
        num=0
        for row in rows[1:]:
            elems = row.strip().split(" ")
            rc = float(elems[1])
            paths[num] = (rc, map(lambda z: int(z), elems[4:]))
            num+=1
            #print "Found path: "+paths[0]+" "+paths[0]
        sys.exit(0)
    else:# we are the parent
        os.waitpid(childid, 0)
    print paths;
    return paths


# Convert a path in G to a column of the master problem
def pathToColumn(path,s,t,A):
    column = []
    cost = 0.0
    for i,j in path:
        arc = A.select(i,j,'*')  # Dirty hack
        _,_,c = arc[0]
        cost += c
        if i < s:
            column.append(i)
    return cost,column


# Build an initial set of paths
def heuristicPaths(N,A,s,t):
    Cs = []
    ############ TODO
    ###### Cs must be a list of tuples made by (cost, column)
    ###### where column is a list of arcs
    return Cs


def solveRestrictedMaster(N,A,s,t,Cs):
    model = Model("RestrictedMaster")
    model.setParam(GRB.Param.Presolve,   0)
    model.setParam(GRB.Param.Method,     1)
    model.setParam(GRB.Param.OutputFlag, 0)
    # Add empty cover constraints
    cover={}
    for i in N:
        if i != s and i != t:
            cover[i] = model.addConstr( 0.0, GRB.GREATER_EQUAL, 1.0)
            # TRY THIS: cover[i] = model.addConstr( 0.0, GRB.EQUAL, 1.0)
    # Update model to integrate new constraints
    model.update()
    # Add node variables
    idx = 0  # Index for tracking columns (go back to paths afterward)
    y = {}
    for cost, column in Cs:
        y[idx] = model.addVar(
             ############## TODO: complete this call
                            )
        idx += 1
    # The objective is to minimize the total costs
    model.modelSense = GRB.MINIMIZE
    model.optimize()
     # Check Solver Status
    if model.status != GRB.status.OPTIMAL:
        print 'Shortest Path - ERROR:',model.status
    print "Relaxed Restricted Master: "+str(model.objVal)



def columnGeneration(N,A,s,t,Cs):
    model = Model("RestrictedMaster")
    model.setParam(GRB.Param.Presolve,   0)
    model.setParam(GRB.Param.Method,     1)
    model.setParam(GRB.Param.OutputFlag, 0)
    # Add dummy cover constraints
    cover = {}
    for i in N:
        if i != s and i != t:
            cover[i] = model.addConstr( 0.0, GRB.GREATER_EQUAL, 1.0)
            # TRY THIS: cover[i] = model.addConstr( 0.0, GRB.EQUAL, 1.0)
    # Update model to integrate new constraints
    model.update()
    # Add node variables
    idx = 0  # Index for tracking columns (go back to paths afterward)
    y = {}
    for column in Cs:
        y[idx] = model.addVar(obj=1.0, column=Column([1 for i in column], [cover[i] for i in column]))
        idx += 1
    # The objective is to minimize the total costs
    model.modelSense = GRB.MINIMIZE
    # Start Column Generation Loop
    while True:
        # Optimize
        model.optimize()
        # Check Solver Status
        if model.status != GRB.status.OPTIMAL:
            print 'Shortest Path - ERROR:',model.status
            exit(0)
        # Get Dual Multipliers
        pi = model.getAttr("Pi", cover)
        # Set the new cost on the arcs

        B = []
        ######## TODO: add arc ###################################
        # for i,j,c in A:
            #B += []
        ######## END TODO #######################################
        B = tuplelist(B)

        # Solve the pricing subproblem
        paths = pricingProblem(N,B,s,t)
        # paths = resourceConstrainedPricingProblem(N,B,s,t)

        ########### TODO: check if the path/column has to be added
        ########### and when the column generation process has to terminate
        has_to_terminate=False;
        for rc, path in paths.values():
            if True: ###### TODO change here
                # Build colum from path
                column = pathToColumn(path,s,t)
                Cs += [(cost,column)]
                y[idx] = model.addVar(
                                      ############## TODO: complete this call
                                      ############## you can copy it from your implementation in
                                      ############## solveRestrictedMaster
                                      )
                idx += 1

        if has_to_terminate:
            print "Leave column generation loop"
            break
    return model

def heuristicIntegerSolution(model):
    # Consider the variable as integer
    for y in model.getVars():
        y.setAttr("VType", GRB.BINARY)
    model.setParam(GRB.Param.OutputFlag, 1)
    model.update()
    # Resolve the integer problem
    model.optimize()
    # Print the optimal solution
    y_bar = model.getAttr('X', y)
    for i in y_bar:
        if y_bar[i] > 0.99:
            print Cs[i]
    # Return optimal value and path
    print model.objVal


if __name__ == '__main__':
    tmpdir = "tmp-"+str(os.getpid())
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    # Set of nodes and arcs
    N,A,s,t = readDataNoResource(sys.argv[1])
    # N,A,k,s,t = readDataWithResource(sys.argv[1])
    # Arcs as "tuplelist"
    A = tuplelist(A)
    # Set of nodes
    print "Number of nodes:", len(N), " - Number of arcs:", len(A)
    # Initial pool of columns
    Cs = heuristicPaths(N,A,s,t)

    ### TODO: only needed by exercise 3 ##########################
    solveRestrictedMaster(N,A,s,t,Cs)
    exit(0)
    #### END TODO ######################################

    # Solve problem by using a column generation heuristic
    model=columnGeneration(N,A,s,t,Cs)
    heuristicIntegerSolution(model)
