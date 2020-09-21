function Selection(W,time,X,chromosomes,nonelite,agents,model="degroot",lambda=nothing,bounded=false,delta=nothing,power=2,min=nothing,max=nothing)
#Extract initial opinions
  initial=X[:,1]
  #Identify row where model-level evaluation will be stored
  model_level=agents+1
  #Populate value of objective function for each chromosome by rows
  objective=Array{Float64}(undef,model_level,chromosomes)
  #Loop through all chromosomes
  for k in 1:chromosomes
    #Generate predicted opinions for chromosome k
    Xhat=Opinions(W[k],time,initial,model,lambda,bounded,delta)
    #Calculate value of objective function by rows for chromosome k
    objective[1:agents,k]=PowerDeviation(X,Xhat,power,min,max)
  end
  #Sum row-level value of objective function to find model-level value of objective function
  objective[model_level,:]=sum(objective[1:agents,:],dims=1)
  #Identify index of elite chromosome
  elite_index=findmin(objective[model_level,:])[2][1]
  #Identify index of worse chromosome
  worst=findmax(objective[model_level,:])[2][1]
  #Check whether elite chromosome has changed from previous iteration
  if elite_index!=5
    #Populate list of chromosome indices
    indices=[1:1:chromosomes;]
    #Remove elite chromosome index from list of indices
    deleteat!(indices,elite_index)
    #Add index of elite chromosome to end of list of indices
    push!(indices,elite_index)
    #Reorder chromosomes in W so elite chromosome is last
    W=W[indices]
    #Reorder values of objective function to match new order of chromosomes
    objective=objective[:,indices]
  end
  #Store current best deviation
  best=objective[model_level,chromosomes]
  #Populate array of chromosome with lowest value of objective function by row
  best_row=fill(0,agents)
  #Loop through all agents
  for i in 1:agents
    #Identify minimum value of objective function for row i
    best_row[i]=findmin(objective[i,:])[2][1]
  end
  swap="FALSE"
  #Check for lower value of objective function in nonelite chromosome in any row
  if best_row!=fill(chromosomes,agents)
    #Populate weight matrix for attempting row-swapping
    W_swap=deepcopy(W)
    #Loop through all rows
    for i in 1:agents
      #Check for a better row i in nonelite chromosomes
      if best_row[i]!=chromosomes
        #W not defined
        #global W=W
        #Replace row i in elite chromosome with row i from chromosome with lower value of objective function
        W_swap[chromosomes][i,:]=W[best_row[i]][i,:]
        #Replace row i of chromosome with lower value of objective function woth row i from elite chromosome
        W_swap[best_row[i]][i,:]=W[chromosomes][i,:]
      end
    end
    #Generate predicted opinions for swapped elite chromosome
    Xhat_swap=Opinions(W_swap[chromosomes],time,initial,model,lambda,bounded,delta)
    #Calculate model-level value of objective function for swapped elite chromosome
    objective_swap=sum(PowerDeviation(X,Xhat_swap,power,min,max))
    #Check if swapped elite chromosome has lower value of objective function than original elite chromosome
    if objective_swap<best
      swap="TRUE"
      #Replace W with swapped version
      W=W_swap
      #Store deviation of swapped chromosome as deviation of best chromosome
      best=objective_swap
    end
  end
  return W,best,worst,swap
end
