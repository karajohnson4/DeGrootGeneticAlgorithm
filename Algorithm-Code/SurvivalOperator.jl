function Survival(W,Wstar,time,X,chromosomes,nonelite,agents,model="degroot",lambda=nothing,bounded=false,delta=nothing,power=2,min=nothing,max=nothing)
  #Extract initial opinions
  initial=X[:,1]
  #Identify row where model-level evaluation will be stored
  model_level=agents+1
  #Populate value of objective function for each nonelite chromosome prior to current iteration by rows
  objective=Array{Float64}(undef,model_level,nonelite)
  #Populate value of objective function for each nonelite chromosome after current iteration by rows
  objective_star=Array{Float64}(undef,model_level,nonelite)
  #Loop through all nonelite chromosomes
  for k in 1:nonelite
    #Generate predicted opinions for chromosome k from prior to current iteration
    Xhat=Opinions(W[k],time,initial,model,lambda,bounded,delta)
    #Calculate value of objective function by rows for chromosome k from prior to current iteration
    objective[1:agents,k]=PowerDeviation(X,Xhat,power,min,max)
    #Generate predicted opinions for chromosome k from after current iteration
    Xhat=Opinions(Wstar[k],time,initial,model,lambda,bounded,delta)
    #Calculate value of objective function by rows for chromosome k from prior to current iteration
    objective_star[1:agents,k]=PowerDeviation(X,Xhat,power,min,max)
  end
  #Sum row-level value of objective function to find model-level value of objective function for chromosomes prior to current iteration
  objective[model_level,:]=sum(objective[1:agents,:],dims=1)
  #Sum row-level value of objective function to find model-level value of objective function for chromosomes after current iteration
  objective_star[model_level,:]=sum(objective_star[1:agents,:],dims=1)
  #Populate list of chromosomes where row-swapping is attempted
  attempted=[0,0]
  #Populate chromosomes for attempted row-swapping
    Wswap=deepcopy(Wstar)
    #populate chromosomes for rejected rows
    Wswap_rejected=deepcopy(W)
  #Loop through all chromosomes
  for k in 1:nonelite
    #Check if new chromosome k is not an improvement
    if objective_star[model_level,k]>objective[model_level,k]
      #Store modified chromosome k
      store_chromosome=Wstar[k]
      #Store values of objective function for modified chromosome k
      store_objective=objective_star[:,k]
      #Replace modified chromosome k with chromosome k prior to last iteration
      Wstar[k]=W[k]
      #Replace values of objective function for modified chromosome k with values of objective function for chromosome k prior to last iteration
      objective_star[:,k]=objective[:,k]
      #Replace chromosome k prior to last iteration with rejected chromosome k from current iteration
      W[k]=store_chromosome
      #Replace values of objective function for chromosome k prior to last iteration with values of objective function for rejected chromosome k from current iteration
      objective[:,k]=store_objective
    end
    #Loop through all rows
    for i in 1:agents
      #Check whether row i in rejected chromosome k is better than corresponding row in offspring chromosome
      if objective[i,k]<objective_star[i,k]
        #Check if chromosome k needs to be added to list of chromosomes where row-swapping is attempted
        if in(k,attempted)==false
          #Add k to list of chromosomes where row-swapping is attempted
          push!(attempted,k)
        end
        #Swap rejected row i from offspring chromosome k into row i of rejected chromosome k
        Wswap_rejected[k][i,:]=Wswap[k][i,:]
        #Swap better row i from parent chromosome k into row i of row-swapped chromosome k
        Wswap[k][i,:]=W[k][i,:]
      end
    end
  end
  #Remove initial values from list of chromosomes where row-swapping is attempted
  deleteat!(attempted,(1,2))
  #Loop through all chromosomes where row-swapping was attempted
  for k in attempted
    #Generate predicted opinions for chromosome k from after current iteration
    Xhat=Opinions(Wswap[k],time,initial,model,lambda,bounded,delta)
    #Calculate model-level value of objective function for chromosome k from prior to current iteration
    objective_swap=sum(PowerDeviation(X,Xhat,power,min,max))
    #Check if row-swapping is an improvment
    if objective_swap<objective_star[model_level,k]
      #Replace parent chromosome k prior to row-swapping with rejected chromosome k
      W[k]=Wswap_rejected[k]
      #Replace offspring chromosome k prior to row-swapping with swapped offspring chromosome
      Wstar[k]=Wswap[k]
    end
  end
  return Wstar
end
