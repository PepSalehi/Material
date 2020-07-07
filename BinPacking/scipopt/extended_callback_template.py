from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum



def generate_initial_patterns(s, B):
    # Generate initial patterns with one size for each item width
    t = []
    m = len(s)
    for i in range(m):
        pat = [0]*m  # vector of number of orders to be packed into one bin
        pat[i] = int(B/s[i])
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
        ### Add here the model
        ###############################################################
        
        # Adding the column to the master problem
        if Condition:
            ###
            ### Add here
            ###
            
            # Storing the new variable in the pricer data.
            self.data['patterns'].append(newPattern)
            self.data['var'].append(newVar)

        return {'result':SCIP_RESULT.SUCCESS}

    # The initialisation function for the variable pricer to retrieve the transformed constraints of the problem
    def pricerinit(self):
        for i, c in enumerate(self.data['cons']):
            self.data['cons'][i] = self.model.getTransformedCons(c)


def SolveBinPacking():
    # create solver instance
    s = Model("BinPacking")

    s.setPresolve(0)

    # creating a pricer
    pricer = PatternPricer()
    s.includePricer(pricer, "BinPackingPricer", "Pricer to identify new patterns")

    
    # adding the initial variables
    PatternVars = []
    varNames = []
    varBaseName = "Pattern"
    patterns = []

    initialCoeffs = []
    for i in range(len(widths)):
        varNames.append(varBaseName + "_" + str(i))
        PatternVars.append(s.addVar(varNames[i], obj = 1.0))

    # adding a linear constraint for the knapsack constraint
    demandCons = []
    

    # Setting the pricer_data for use in the init and redcost functions
    pricer.data = {}
    pricer.data['var'] = cutPatternVars
    pricer.data['cons'] = demandCons
    pricer.data['widths'] = widths
    pricer.data['demand'] = demand
    pricer.data['rollLength'] = rollLength
    pricer.data['patterns'] = patterns

    # solve problem
    s.optimize()

    # print original data
    printWidths = '\t'.join(str(e) for e in widths)
    print('\nInput Data')
    print('==========')
    print('Roll Length:', rollLength)
    print('Widths:\t', printWidths)
    print('Demand:\t', '\t'.join(str(e) for e in demand))

    # print solution
    widthOutput = [0]*len(widths)
    print('\nResult')
    print('======')
    print('\t\tSol Value', '\tWidths\t', printWidths)
    for i in range(len(pricer.data['var'])):
        rollUsage = 0
        solValue = round(s.getVal(pricer.data['var'][i]))
        if solValue > 0:
            outline = 'Pattern_' + str(i) + ':\t' + str(solValue) + '\t\tCuts:\t '
            for j in range(len(widths)):
                rollUsage += pricer.data['patterns'][i][j]*widths[j]
                widthOutput[j] += pricer.data['patterns'][i][j]*solValue
                outline += str(pricer.data['patterns'][i][j]) + '\t'
            outline += 'Usage:' + str(rollUsage)
            print(outline)

    print('\t\t\tTotal Output:\t', '\t'.join(str(e) for e in widthOutput))







if __name__ == '__main__':

    s, B = BinPackingExample()
    ffd = FFD(s, B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\nBin Packing, column generation:")
    bins = solveBinPacking(s, B)
    print(len(bins), "rolls:")
    print(bins)
