function PowerDeviation(X,Xhat::Array{Float64,2},power::Int64=2,min=nothing,max=nothing)
  #Calculates the absolute deviation raised to the specified power for predicted opinions and observed opinions at either the agent or model level
  # X: N x T matrix of observed opinions
  # Xhat: N x T matrix of predicted opinions
  #Count number of agents and timepoints
  agents,observations=size(X)
  #Count rows and columns of Xhat
  rows,columns=size(Xhat)
  #Check number of rows in Xhat matches number of rows (agents) in X
  if rows!=agents
    error("PowerDeviation:number of rows in X and Xhat do not match")
  end
  #Check number of columns in Xhat matches number of columns (timepoints) in X
  if columns!=observations
    error("PowerDeviation: number of columns in X and Xhat do not match")
  end
  #Loop through all agents
  for i in 1:agents
    #Loop through all timepoints
    for t in 1:observations
      #Check if observation t for agent i is missing
      if ismissing(X[i,t])
        #Replace missing observed opinion for agent i at timepoint t with predicted opinion for agent i at timepoint t
        X[i,t]=Xhat[i,t]
      end
    end
  end
  #Calculate deviation for each agent and timepoint
  #deviation=(abs.(X-Xhat)).^power
  deviation=(abs.(Scale(X,min,max,true).-Scale(Xhat,min,max,true))).*abs.(X-Xhat)
  #Sum deviations across rows and take powerth root
  deviation=((sum(deviation,dims=2))./(agents*(observations-1)))#.^(1/power)
  #Return deviation(s)
  return deviation
end
