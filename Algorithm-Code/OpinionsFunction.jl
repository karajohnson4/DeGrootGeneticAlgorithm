function Opinions(W::Array{Float64,2},time::Int64,initial,model::String="degroot",lambda=nothing,bounded::Bool=false,delta=nothing)
  # Calculates estimated opinions for the specified opinion diffusion model given initial opinions and model parameters
  # W: N x N array of arrays of weights where each row is an array
  # time: total number of time points including initial (t=0)
  # initial: N x 1 vector of predicted initial opinions
  # model: type of opinion diffusion model ("degroot","decay","initial decay"), "degroot" is the default
  # lambda: optional decay parameter in (0,1)
  # bounded.confidence: logical, if TRUE bounded confidence is applied to the model, weight is distributed proportionally to other elements
  # delta: optional bounded confidence parameter in (0,1)

  #Count number of agents
  agents=size(W,1)
  #check W is square
  if size(W,2)!=agents
   error("Opinions: W is not square")
  end
  #Loop through all agents
  for i in 1:agents
    #Check row i sums to 1
    if isapprox(1,sum(W[i,:]))==false
      newsum=sum(W[i,:])
      println(W[i,:])
      error("Opinions: row $i of W sums to $newsum")
    end
  end
  #Check number of initial opinions is the same as the number of agents
  if length(initial)!=agents
    error("Opinions: W and initial do not have the same number of rows")
  end
  #Check for valid model type
  if model in ["degroot","decay","initial decay"]==false
    error("Opinions: Invalid model type")
  end
  #Decay or initial decay models
  if model in ["decay","initial decay"]
    #Check if decay parameter (lambda) is missing
    if lambda==nothing
      warn("Opinions: $model selected with no decay parameter (lambda) specified, degroot was used")
      #Change model to degroot
      model="degroot"
    #Check if decay parameter is specified but not between 0 and 1
    elseif lambda<0 || lambda>1
      error("Opinions: decay parameter (lambda) must be between 0 and 1")
    end
  end
  #Check for unnecessary decay parameter (lambda) for degroot model
  if model=="degroot" && lambda!=nothing
    warn("Opinions: decay parameter (lambda) is unnnecessary for degroot model and was ignored")
    #Overwrite lambda with no value for degroot model
    lambda=nothing
  end
  #Check for unnecessary bounded confidence parameter (delta)
  if bounded==false && delta!=nothing
      warn("Opinions: bounded confidence parameter (delta) is unnnecessary for model without bounded confidence and was ignored")
  end
  #Populate matrix of predicted opinions
  Xhat=rand(agents,time)
  #Replace first column of Xhat with initial opinions
  Xhat[:,1]=initial
  #Check for bounded confidence model
  if bounded==true
    #Check bounded confidence parameter (delta) is specified
    if delta==nothing
      warn("Opinions: bounded confidence selected with no bounded confidence parameter (delta) specified, bounded confidence ignored")
      #Ignore request for bounded confidence
      bounded=false
    #Check if decay parameter specified but not between 0 and 1
    elseif delta<0 || delta>1
      error("Opinions: bounded confidence parameter (delta) must be between 0 and 1")
    end
    if model=="decay"
      #Loop through all but last timepoint
      for t in 1:(time-1)
        #Populate Wbounded
        Wbounded=W
        #Loop through all agents (columns)
        for j in 1:agents
          #Loop through all agents past agent j (columns)
          for i in (j+1):agents
            #Check if agents i and j are outside bound for influenced
            if abs(Xhat[i,t]-Xhat[j,t])>delta
              #Force weight agent i places on agent j to 0
              Wbounded[i,j]=0
              #Scale row i so it still sums to 1
              Wbounded[i,:]=Wbounded[i,:]/sum(Wbounded[i,:])
              #BANDAID
              #Check if Wbounded contains any nonzero elements
              if sum(Wbounded[i,:])==0
                Wbounded[i,:]=zeros(1,agents)
                Wbounded[i,i]=1
              end
              #Force weight agent j places on agent i to 0
              Wbounded[j,i]=0
              #Scale row j so it still sums to 1
              Wbounded[j,:]=Wbounded[j,:]/sum(Wbounded[j,:])
              #BANDAID
              #Check if Wbounded contains any nonzero elements
              if sum(Wbounded[j,:])==0
                Wbounded[j,:]=zeros(1,agents)
                Wbounded[j,j]=1
              end
            end
          end
        end
        #Populate Xhat with opinions for timepoint t+1
        Xhat[:,t+1]=((1-lambda^(t-1))*I+lambda^(t-1)*Wbounded)*Xhat[:,t]
      end
    #Initial decay model
    elseif model=="initial decay"
      #Loop through all but last timepoint
      for t in 1:(time-1)
        #Populate Wbounded
        Wbounded=W
        #Loop through all agents (columns)
        for j in 1:agents
          #Loop through all agents past agent i (rows)
          for i in (j+1):agents
            #Check if agents i and j are outside bound for influenced
            if abs(Xhat[i,t]-Xhat[j,t])>delta
              #Force weight agent i places on agent j to 0
              Wbounded[i,j]=0
              #Scale row i so it still sums to 1
              Wbounded[i,:]=Wbounded[i,:]/sum(Wbounded[i,:])
              #Force weight agent j places on agent i to 0
              Wbounded[j,i]=0
              #Scale row j so it still sums to 1
              Wbounded[j,:]=Wbounded[j,:]/sum(Wbounded[j,:])
            end
          end
        end
        #Populate Xhat with opinions for timepoint t+1
        Xhat[:,t+1]=(lambda^t)*initial+((1-lambda^t)*Wbounded)*Xhat[:,t]
      end
    #DeGroot model
    else
      #Loop through all but last timepoint
      for t in 1:(time-1)
        #Populate Wbounded
        Wbounded=W
        #Loop through all agents (columns)
        for j in 1:agents
          #Loop through all agents past agent i (rows)
          for i in (j+1):agents
            #Check if agents i and j are outside bound for influenced
            if abs(Xhat[i,t]-Xhat[j,t])>delta
              #Force weight agent i places on agent j to 0
              Wbounded[i,j]=0
              #Scale row i so it still sums to 1
              Wbounded[i,:]=Wbounded[i,:]/sum(Wbounded[i,:])
              #Force weight agent j places on agent i to 0
              Wbounded[j,i]=0
              #Scale row j so it still sums to 1
              Wbounded[j,:]=Wbounded[j,:]/sum(Wbounded[j,:])
            end
          end
        end
        #Populate Xhat with opinions for timepoint t+1
        Xhat[:,t+1]=(lambda^t)*initial+((1-lambda^t)*Wbounded)*Xhat[:,t]
      end
    end
  #Models without bounded confidence
  else
    #Decay model
    if model=="decay"
      #Loop through all but last timepoint
      for t in 1:(time-1)
        #Populate Xhat with opinions for timepoint t+1
        Xhat[:,t+1]=((1-lambda^(t-1))*I+lambda^(t-1)*W)*Xhat[:,t]
      end
    #Initial decay model
    elseif model=="initial decay"
      #Loop through all but last timepoint
      for t in 1:(time-1)
        #Populate Xhat with opinions for timepoint t+1
        Xhat[:,t+1]=(lambda^t)*initial+((1-lambda^t)*W)*Xhat[:,t]
      end
    #DeGroot model
    else
      #Loop through all but last timepoint
      for t in 1:(time-1)
        #Populate Xhat with opinions for timepoint t+1
        Xhat[:,t+1]=W*Xhat[:,t]
      end
    end
  end
  return Xhat
end
