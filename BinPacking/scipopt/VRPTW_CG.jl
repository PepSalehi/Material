include("VRPTW_data.jl")
using JuMP
using GLPKMathProgInterface

#Parameters
EPS= 0.0001;    #error margin allowed for rounding

#Read Instance
(n, cust,m,Q,demand,twStart,twEnd,cost,timeCost, routes, costRoutes)= VRPTW_data.createInstance();

# ----------------------------------------------
# Setup the master problem with the initial data
#-----------------------------------------------
# Model for the master problem, parameter ensures that CPLEX does not produce output
VRPTWMaster = Model(solver= GLPKSolverLP())
#Defining an empty array of variables
y = Variable[]
#Defining empty constraints
@constraint(VRPTWMaster, useVehicles, 0 <= m)
@constraint(VRPTWMaster, serveCustomer[i=1:cust], 0 >= 1)

#Adding the  intial routes from the 'routes' array from VRPTW_data.jl file
for r = 1:size(routes,1)
    touchedConstraints = ConstraintRef[]
    vals = Float64[]

    for i = 1:cust
        push!(touchedConstraints, serveCustomer[i])
        push!(vals, routes[r,i])
    end
    push!(touchedConstraints, useVehicles)
    push!(vals, 1)

    routeCost = costRoutes[r]
    ##Add the variable to the model
    @variable(
            VRPTWMaster,                           # Model to be modified
            yNew >= 0,                             # New variable to be added
            objective=routeCost,                   # cost coefficient of new varaible in the objective
            inconstraints=touchedConstraints ,     # constraints to be modified
            coefficients=vals                      # the coefficients of the variable in those constraints
           )
    push!(y, yNew) # Pushing the new variable in the array of variables
    name = string("y[", size(y,1),"]")
    setname(yNew, name) #Set the name of the variable
end

# ----------------------------------------------
# Setup the pricing problem
#-----------------------------------------------
# Model for the pricing problem, parameter ensures that CPLEX does not produce output
pricingProb=Model(solver = GLPKSolverMIP(presolve=true)) # Model for the subproblem
##### TODO: Define the constraints for the Subproblem ####


# ----------------------------------------------
# Start of Column Generation
#-----------------------------------------------
iteration = 1
done = false

while !done
    # Solve the master problem
    solve(VRPTWMaster)

    #Collect the dual variables
    lambdaZero = getdual(useVehicles)
    lambda = getdual(serveCustomer)

    ### TODO: Change objective of Pricing problem
    solve(pricingProb)

    ### TODO: Determine new value of the boolean 'done'

    if !done
        ### TODO: Add new variable to master
    end

    iteration = iteration + 1
end
