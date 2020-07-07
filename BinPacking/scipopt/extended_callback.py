# Marco Chiarandini
# Script not yet verified (not working yet)

from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
from data import BinPackingExample, Lister, FFD


def generate_initial_patterns(s, B):
    # Generate initial patterns with one size for each item width
    t = []
    m = len(s)
    for i in range(m):
        pat = [0]*m  # vector of number of orders to be packed into one bin
        pat[i] = 1
        t.append(pat)
    return t


class PatternPricer(Pricer):

    # The reduced cost function for the variable pricer
    def pricerredcost(self):

        # Retrieving the dual solutions
        dualSolutions = []
        for i, c in enumerate(self.data['cons']):
            dualSolutions.append(self.model.getDualMultiplier(c))

        # Building a MIP to solve the subproblem
        subMIP = Model("BinPacking-Sub")

        # Turning off presolve
        subMIP.setPresolve(SCIP_PARAMSETTING.OFF)

        # Setting the verbosity level to 0
        subMIP.hideOutput()

        PatternVars = []
        varNames = []
        varBaseName = "bin"

        ###############################################################
        # Add here the model
        s = self.data["sizes"]
        I = range(len(s))
        subMIP.setMaximize       # maximize
        y = {}
        for i in I:
            y[i] = subMIP.addVar(lb=0, vtype="B", name="y(%s)" % i)

        subMIP.addCons(quicksum(s[i]*y[i] for i in I) <= B, "Capacity")

        subMIP.setObjective(quicksum(dualSolutions[i]*y[i] for i in I), "maximize")

        subMIP.hideOutput()  # silent mode
        subMIP.optimize()
        pat = [round(subMIP.getVal(y[i])) for i in y]
        # return subMIP.getObjVal(), pat
        ###############################################################

        # Adding the column to the master problem
        if 1-subMIP.getObjVal() < -1e-08:
            ###############################################################
            # Add here
            currentNumVar = len(self.data['var'])

            # Creating new var; must set pricedVar to True
            newVar = self.model.addVar("NewPattern_" + str(currentNumVar),
                                       vtype="C", obj=1.0, pricedVar=True)

            # Adding the new variable to the constraints of the master problem
            newPattern = []
            for i, c in enumerate(self.data['cons']):
                coeff = round(subMIP.getVal(y[i]))
                self.model.addConsCoeff(c, newVar, coeff)

                newPattern.append(coeff)

            ###############################################################
            # Storing the new variable in the pricer data.
            self.data['patterns'].append(newPattern)
            self.data['var'].append(newVar)

        return {'result': SCIP_RESULT.SUCCESS}

    # The initialisation function for the variable pricer to retrieve the transformed constraints of the problem
    def pricerinit(self):
        for i, c in enumerate(self.data['cons']):
            self.data['cons'][i] = self.model.getTransformedCons(c)


def SolveBinPacking(s, B):
    # create solver instance
    sp = Model("BinPacking")

    sp.setPresolve(0)

    # creating a pricer
    pricer = PatternPricer()
    sp.includePricer(pricer, "BinPackingPricer", "Pricer to identify new patterns")

    # adding the initial variables
    PatternVars = []
    varNames = []
    varBaseName = "Pattern"
    patterns = []

    I = range(len(s))
    initialCoeffs = []
    for i in I:
        varNames.append(varBaseName + "_" + str(i))
        PatternVars.append(sp.addVar(varNames[i], vtype="B", obj=1.0))

    #########################################################################################
    # adding a linear constraint for the knapsack constraint
    demandCons = []
    for i in I:
        #numWidthsPerRoll = float(int(rollLength/widths[i]))
        demandCons.append(sp.addCons(PatternVars[i] >= 1,
                                     separate=False, modifiable=True))
        newPattern = [0]*len(s)
        newPattern[i] = 1
        patterns.append(newPattern)
    #######################################################################################
    # Setting the pricer_data for use in the init and redcost functions
    pricer.data = {}
    pricer.data['var'] = PatternVars
    pricer.data['cons'] = demandCons
    pricer.data['sizes'] = s
    pricer.data['BinCap'] = B
    pricer.data['patterns'] = patterns

    # solve problem
    sp.optimize()

    sp.writeSol(sp.getSols()[0], "sol.sol")
    # print original data
    printSizes = '\t'.join(str(e) for e in s)
    print('\nInput Data')
    print('==========')
    print('Bin capacity:', B)
    print('Sizes:\t', printSizes)
    #print('Demand:\t', '\t'.join(str(e) for e in demand))

    # print solution
    sizesOutput = [0]*len(s)
    print('\nResult')
    print('======')
    print('\t\tSol Value', '\tSizes\t', printSizes)
    for i in range(len(pricer.data['var'])):
        binUsage = 0
        solValue = round(sp.getVal(pricer.data['var'][i]))
        if solValue > 0:
            outline = 'Pattern_' + str(i) + ':\t' + str(solValue) + '\t\tItems:\t '
            for j in range(len(s)):
                binUsage += pricer.data['patterns'][i][j]*s[j]
                sizesOutput[j] += pricer.data['patterns'][i][j]*solValue
                outline += str(pricer.data['patterns'][i][j]) + '\t'
            outline += 'Usage:' + str(binUsage)
            print(outline)

    print('\t\t\tTotal Output:\t', '\t'.join(str(e) for e in sizesOutput))


if __name__ == '__main__':

    s, B = BinPackingExample()
    s, B = Lister()
    ffd = FFD(s, B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\nBin Packing, column generation:")
    SolveBinPacking(s, B)
