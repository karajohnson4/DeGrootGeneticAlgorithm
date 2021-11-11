function GA(X,W=nothing,model="degroot",lambda=nothing,bounded=false,delta=nothing,chromosomes=5,A=nothing,fixed=nothing,initial=true,probb=0.01,factorb=2,maxb=0.2,max_iterb=2000,probc=0.2,factorc=0.5,minc=0,max_iterc=2000,probm=0.2,factorm=0.5,minm=0.01,max_iterm=2000,sigma=1,factors=0.5,mins=0.001,max_iters=2000,power=2,max_iter=10000000,print_iter=1000,min_improve=0.0001,min_dev=0.001,reintroduce="elite",max_iterr=1500,min=nothing,max=nothing)
  #X: N x T matrix of observed opinions
  #W: optional array of N x N initial chromosomes
  #model: type of opinion diffusion model ("degroot","decay","initial decay"), "degroot" is the default
  #lambda: opional initial value for decay parameter (lambda) for "decay" or "initial decay" models
  #bounded: logical, if true bounded confidence model is implimented
  #delta: optional initial value for bounded confidenced parameter (delta)
  #chromosomes: optional number of chromosomes, defaults to 5 if no chromosomes are specified
  #A: optional adjacency matrix for generating additional chromosomes
  #fixed: optional matrix with 3 columns specifying fixed values, entered as position i,j and element w
  ##initial: logical, if FALSE exclude time t=0 from calculation
  #probb: blending probability between 0 and 1
  #probc: crossover probability between 0 and 1
  #probm: mutation probability between 0 and 1
  #sigma: mutation parameter with sigma^2 being the variance parameter for a normal distibution with mean 0
  #power: power to use for objective function
  #max_iter: maximum number of iterations
  #print:

  #Count number of agents and timepoints
  agents,time=size(X)
  #Store initial values from observed opinions
  initial=X[:,1]
  #Populate identity matrix
  #diagonal=Matrix{Float64}(I,agents,agents)
  #Check number of chromosomes requested is odd and at least 5
  if mod(chromosomes,2)!=1 || chromosomes<5
    error("GA: number of chromosomes (chromosomes) must be odd and at least 5")
  end
  #Check if no initial chromosomes are specified
  if W==nothing
    #Set number of user-specified chromosomes to 0
    user=0
  else
    #Count number of user-specified chromosomes
    user=size(W,1)
    #Loop through all user-specified chromosomes
    for k in 1:user
      #Identify number of rows and columns in chromosome i
      rows,columns=size(W[k])
      #Check chromosome i is square
      if rows!=columns
        error("GA: user-specified chromosome $k is not square")
      end
      #Check chromosome size agrees with number of agents
      if rows!=agents
        error("GA: dimensions of user-specified chromosome $k does not agree with number of agents")
      end
      #Loop through all agents (rows)
      for i in 1:agents
        #Check sum of row i is approximately 1
        if isapprox(sum(W[k][i,:]),1)==false
          error("GA: row $i in user-specified chromosome $k does not sum to 1")
        end
      end
    end
  end
  #Check for valid model type
  if model in ["degroot","decay","initial decay"]==false
    error("GA: Invalid model type")
  end
  #Check for unnecessary decay parameter (lambda) for degroot model
  if model=="degroot" && lambda!=nothing
    warn("GA: decay parameter (lambda) is unnnecessary for degroot model and was ignored")
    #Overwrite decay parameter (lambda) as nothing
    lambda=nothing
  end
  #Check for unnecessary bounded confidence parameter (delta) for model without bounded confifdence
  if bounded==false && delta!=nothing
    warn("GA: bounded confidence parameter (delta) is unnnecessary for model without bounded confidence and is ignored")
    #Overwrite bounded confidence parameter (delta) as nothing
    delta=nothing
  end
  #Check value for decay parameter (lambda) is between 0 and 1 if specified
  if lambda!=nothing && (lambda<=0 || lambda=>1)
    warn("GA: decay parameter (lambda) not between 0 and 1, initial value will be randomly generated")
    #Overwrite decay parameter (lambda) as nothing
    lambda=nothing
  end
  #Check value for bounded confidence parameter (delta) is between 0 and 1 if specified
  if delta!=nothing && (delta<=0 || delta>=1)
    warn("GA: bounded confidence parameter (delta) not between 0 and 1, initial value will be randomly generated")
    #Overwrite bounded confidence parameter (delta) as nothing
    delta=nothing
  end
  #Check if adjacency matrix A is specified
  if A!=nothing
    #Loop through all values in adjacency matrix
    for i in 1:length(A)
      #Check all values are either 0 or 1
      if A[i]!=0 && A[i]!=1
        error("GA: element $i in adjacency matrix A must be either 0 or 1")
      end
    end
    #Loop through all user-specified chromosomes
    for k in 1:user
      #Loop through all elements in adjacency matrix A
      for i in 1:(agents^2)
        #Check if element i in adjacency matrix A agrees with user-specified chromosome k
        if W[k][i]>A[i]
          error("GA: element $i in user-specified chromosome $k does not agree with adjacency matrix A")
        end
      end
    end
  #No adjacency matrix A specified
  else
    #Check for at least one user-specified chromosome
    if user==0
      error("GA: either an adjacency matrix (A) or at least one chromosome must be specified")
    end
    #Populate adjacency matrix
    A=zeros(Int8,agents,agents)
    #Loop through all user-specified chromosomes
    for k in 1:user
      #Loop through all elements in chromosome k
      for i in 1:(user^2)
        #check if element i in user-specified chromosome k is nonzero
        if W[k][i]>0
          #Overwrite element i in adjacency matrix A as 1
          A[i]=1
        end
      end
    end
  end
  #Populate matrix of fixed values
  matrix=Array{Union{Missing,Float64}}(missing,agents,agents)
  #Populate modified identity matrix
  diagonal=Array{Float64}(fill(0,agents,agents))
  #Check if any fixed values are specified
  if fixed!=nothing
    #Loop through all fixed values
    for i in 1:size(fixed,1)
      #Identify fixed value i
      value=fixed[i,3]
      #Check if fixed value i is between 0 and 1
      if value<0 || value>1
        error("GA: fixed value $i is not between 0 and 1 inclusive")
      end
    end
    #Loop through all fixed values
    for i in 1:size(fixed,1)
      #Itendify row of fixed value i
      row=trunc(Int,fixed[i,1])
      #Identify column of fixed value i
      column=trunc(Int,fixed[i,2])
      #Identify value of fixed value i
      value=fixed[i,3]
      #Place fixed value i in matrix of fixed values
      matrix[row,column]=value
      #Place fixed value i in modified identity matrix
      diagonal[row,column]=value
    end
  end
  #Loop through all elements in adjacenty matrix
  for i in 1:(agents^2)
    #Check whether element i of adjacency matrix is 0
    if A[i]==0
      #Indicate element i in matrix of fixed values is fixed as 0
      matrix[i]=0
    end
  end
  #Loop through all agents (rows)
  for i in 1:agents
    #Check if agent i has more than 1 connection
    if sum(A[i,:])==1
      #Identify index of connection
      index=findmax(A[i,:])[2]
      #Indicate element index in row i is fixed as 1
      matrix[i,index]=1
    end
  end
  #Initialize sum of fixed values
  sum_fixed=Array{Float64}(undef,agents)
  #Initialize vector of rows with fixed values
  fixed_rows=[0,0]
  #Initialize vector of rows with no fixed values
  variable_rows=[0,0]
  #Loop through all agents (rows)
  for i in 1:agents
    #Find sum of fixed values for agent i
    sum_i=sum(skipmissing(matrix[i,:]))
    #Include sum of fixed values for agent i in vector of sum of fixed values
    sum_fixed[i]=sum_i
    #Include diagonal value in modified identity matrix
    diagonal[i,i]=1-sum_i
    #Check for any fixed values in row i
    if length(collect(skipmissing(matrix[i,:])))>0
      #Indicate row i has fixed values
      push!(fixed_rows,i)
    #No fixed values in row i
    else
      #Indicate row i has no fixed values
      push!(variable_rows,i)
    end
  end
  #Loop through all rows (agents)
  for i in 1:agents
    #Check if agent i places weight on only one agent
    if sum(A[i,:])==1
      #Overwrite value in modified identity matrix
      diagonal[i,:]=A[i,:]
    end
  end
  #Remove initialized values from list of rows with fixed values
  deleteat!(fixed_rows,(1,2))
  #Remove initialized values from list of rows without fixed values
  deleteat!(variable_rows,(1,2))
  #Initialize collection of list of variable elements within each row
  variable_list=fill(Int[],agents)
  #Initialize collection of lists of fixed elements within each row
  fixed_list=fill(Int[],agents)
  #Loop through all agents (rows)
  for i in 1:agents
    #Initialize varfiable elements in row
    variable_row=[0,0]
    #Initialize fixed elements within the rwo
    fixed_row=[0,0]
    #Loop through all agents (columns)
    for j in 1:agents
      #Check if element j for agent i is variable
      if ismissing(matrix[i,j])
        #Add element j to list of variable elements for agent (row) i
        push!(variable_row,j)
      #Element j for agent i is fixed
      else
        #Add element j to list of fixed elements for agent i
        push!(fixed_row,j)
      end
    end
    #Populate variable elements for row i after removing initialized value
    variable_list[i]=deleteat!(variable_row,(1,2))
    #Populate variable elements for row i after removing initialized value
    fixed_list[i]=deleteat!(fixed_row,(1,2))
  end
  #Loop through all agents (rows)
  for i in 1:agents
    #Check if sum of row i excedes 1
    if sum(skipmissing(matrix[i,:]))>1
      error("GA: fixed values for agent $i are too large and will produce non-stochastic matrix")
    end
    #Loop through all agents (columns)
    for j in 1:agents
      #Check fixed value of weight j for agent i agrees with adjacency matrix A
      if A[i,j]==0 && ismissing(matrix[i,j])==false && matrix[i,j]!=0
        error("GA: fixed value of weight $j for agent $i does not agree with adjacency matrix A")
      end
    end
  end
  #Loop through all user-specified chromosomes
  for k in 1:user
    #Loop through all agents (columns)
    for j in 1:agents
      #Loop through all agents (rows)
      for i in 1:agents
        #Check fixed value of weight j for agent i agrees with user-specified chromosome k
        if matrix[i,j]!=W[k][i,j]
          error("GA: fixed value of weight $j for agent $i does not agree with user-specified chromosome $k")
        end
      end
    end
  end
  #Check if any initial values are missing
  if sum(ismissing.(X[:,1]))>0
    error("GA: initial values must not be missing for any agents")
  end
  #Check if less than 5 user-specified chromosomes are provided
  if user<5
    #Check number of chromosomes requested is at least 5
    if chromosomes<5
      error("GA: number of chromosomes (chromosomes) must be at least 5")
    end
  end
  #Check if more than requested number of chromosomes are specified
  if user>chromosomes
    #Check if an even number of chromosomes are specified
    if mod(user,2)==0
      #Set number of chromosomes to be generated to 1
      generate=1
      info("GA: even number of chromosomes provided, one more will be generated")
    #Odd number of user-specified chromosomes
    else
      #Set number of chromosomes to be generated to 0
      generate=0
    end
    #Set number of chromosomes to be total number of chromosomes after generation
    chromosomes=user+generate
  #Check if less than number of requested chromosomes are user-specified
  elseif user<chromosomes
    #Find number of additional chromosomes necessary
    generate=chromosomes-user
  #Number of chromosomes requested same as number of user-specified chromosomes
  else
    generate=0
  end
  #Check whether any chromosomes are to be generated
  if generate>0
    #Save user-specified chromosomes
    W_user=W
    #Populate array of arrays of chromosomes
    W=fill(Array{Float64}(undef,user,user),chromosomes)
    #Loop through all user-specified chromosomes
    for i in 1:user
      W[i]=W_user[i]
    end
    #Ititialize check for user-specified identity matrix
    specified=0
    #Loop through all user-specfied chromosomes
    for i in 1:user
      if W[i]==diagonal
        #global specified+=1
        specified+=1
      end
    end
    #Check whether identity matrix needs to be introduced
    if specified==0
      #Increase count of user-specified chromosomes
      user+=1
      #Add identity matrix to user-specified chromosomes
      W[user]=diagonal
      #Decrease number of additional chromosomes
      generate-=1
    end
    #Check if any fixed values are specified
    if fixed==nothing
      #Loop through all additional chromosomes to be generated
      for k in 1:generate
        #Populate generated chromosome k
        w=rand(agents,agents)
        #Correct generated chromosome k based on adjacency matrix
        w=w.*A
        #Loop through all agents (rows) of generated chromosome k
        for i in 1:agents
          #Rescale so row i sums to 1
          w[i,:]=w[i,:]/sum(w[i,:])
        end
        #Include chromosome k in list of chromosomes
        W[user+k]=w
      end
    #Fixed values are specified
    else
      #Loop through all additional chromosomes to be generated
      for k in 1:generate
        #Populate generated chromosome k
        w=rand(agents,agents)
        #Correct generated chromosome k based on adjacency matrix
        w=w.*A
        #Loop through all agents (rows) of generated chromosome k
        for i in 1:agents
          #Initialize sum of variable values
          sum_variable=sum(w[i,variable_list[i]])
          #loop through all agents (columns)
          for j in 1:agents
            #Check if element j for agent i is fixed
            if ismissing(matrix[i,j])==false
              #Insert fixed element j for agent i
              w[i,j]=matrix[i,j]
            end
          end
          if sum_variable>0
            #Rescale non-fixed elements in row i
            w[i,variable_list[i]]=w[i,variable_list[i]].*((1-sum_fixed[i])/sum_variable)
          end
        end
        #Include chromosome k in list of chromosomes
        W[user+k]=w
      end
    end
  end
  #Identify the number of non-elite chromosomes
  nonelite=chromosomes-1;
  #Initialize iteration counter
  iter=0
  #Initialize iteration counter for blending
  iterb=0
  #Initialize iteration counter for crossover
  iterc=0
  #Initialize iteration counter for mutation
  iterm=0
  #initialize iteration counter for sigma
  iters=0
  #Initialize iteration counter for reintroducing chromosome
  iterr=0
  #Initialize counter for iterations without improvment
  iter_improve=0
  #Initialize old  and new best deviation
  new,old=agents^2,agents^2-1
  #Loop throgh operators
  while true
    #Increase iteration counter by 1
    iter+=1
    #Impliment selection operator and store reordered chromosomes and new best deviation
    W,new,replace,swap=Selection(W,time,X,chromosomes,nonelite,agents,model,lambda,bounded,delta,power,min,max)
    #Check if deviation of elite chromosome is less than minimum deviation
    if new<=min_dev
      #println("Minimum deviation of $min_dev reached after $iter iterations")
      #End loop through operators
      break
    end
    #Check if maximum number of iterations is exceded
    if iter>max_iter
      #println("Maximum number of iterations $max_iter reached")
      #End loop through operators
      break
    end
    #Check for improved elite chromosome
    if new<old
      #Reset counter of iterations with no improvment
      iter_improve=0
      #Reset counter of iterations with no improvment for blending
      iterb=0
      #Reset counter of iterations with no improvment for crossover
      iterc=0
      #Reset counter of iterations with no improvment for mutation
      iterm=0
      #Reset counter of iterations with no improvment for sigma
      iters=0
      #Reset counter of iterations with no improvment for reintroduction of chromosome
      iterr=0
    #Elite chromosome unchanged
    else
      #Increase counter of iterations with no improvment
      iter_improve+=1
      #Increase counter of iterations with no improvment for blending
      iterb+=1
      #Increase counter of iterations with no improvment for crossover
      iterc+=1
      #Increase counter of iterations with no improvment for mutation
      iterm+=1
      #Increase counter of iterations with no improvment for sigma
      iters+=1
      #Increase counter of iterations with no improvment for reintroduction of chromosome
      iterr+=1
    end
    #Check if maximuim number of iterations without improvment for blending operator is reached
    if iterb>max_iterb
      #Multiply probability of blending by specified factor
      probb*=factorb
      #Reset counter of iterations with no improvment for blending operator
      iterb=0
      #Check if new probability of blending excedes specified max
      if probb>maxb
        #Reset probability of blending to specified max
        probb=maxb
      end
      #println("Probability of blending increased to $probb after $max_iterb iterations with no improvment at iteration $iter")
    end
    #Check if maximuim number of iterations without improvment for crossover operator is reached
    if iterc>max_iterc
      #Multiply probability of crossover by specified factor
      probc*=factorc
      #Reset counter of iterations with no improvment for crossover operator
      iterc=0
      #Check if new probability of crossover is below specified min
      if probc<minc
        #Reset probability of crossover to specified min
        probc=minc
      end
      #println("Probability of crossover decreased to $probc after $max_iterc iterations with no improvment at iteration $iter")
    end
    #Check if maximuim number of iterations without improvment for mutation operator is reached
    if iterm>max_iterm
      #Multiply probability of mutation by specified factor
      probm*=factorm
      #Reset counter of iterations with no improvment for mutation operator
      iterm=0
      #Check if new probability of mutation is below specified min
      if probm<minm
        #Reset probability of mutation to specified min
        probm=minm
      end
      #println("Probability of mutation decreased to $probm after $max_iterm iterations with no improvment at iteration $iter")
    end
    #Check if maximuim number of iterations without improvment for sigma is reached
    if iters>max_iters
      #Multiply sigma by specified factor
      sigma*=factors
      #Reset counter of iterations with no improvment for sigma
      iters=0
      #Check if new value of sigma is below specified min
      if sigma<mins
        #Reset sigma to specified min
        sigma=mins
      end
      #println("Sigma decreased to $sigma after $max_iters iterations with no improvment at iteration $iter")
    end
    #Check if maximum number of iterations with no improvment for chromosome reintroduction is reached
    if iterr>max_iterr
      #Reset counter for iterations with no improvment for chromosome reintroduction
      iterr=0
      #Check for elite chromosome reintroduction
      if reintroduce=="elite"
        #Replace worst chromosome with elite chromosome
        W[replace]=W[chromosomes]
        #println("Elite chromosome reintroduced after $max_iterr iterations with no improvment at iteration $iter")
      end
      #Check for identity matrix reintroduction
      if reintroduce=="identity"
        #Replace worst chromosome with identity (or modified identity) matrix
        W[replace]=diagonal
        #println("Identity matrix reintroduced after $max_iterr iterations with no improvment at iteration $iter")
      end
    end
    #Save copy of parent chromosomes
    Wold=deepcopy(W)
    #Check if iteration details should be printed
    if mod(iter,print_iter)==0 || iter==1
      dev_print=new*1000
      #println("Iter: $iter, Dev: $dev_print")
    end
    #Save version of W for blending operator
    Wstar=deepcopy(W)
    #Impliment blending operator
    Wstar=Blending(Wstar,nonelite,agents,probb)
    #Impliment crossover operator
    Wstar=Crossover(Wstar,nonelite,agents,probc,variable_list,fixed_rows,variable_rows)
    #Impliment mutation operator
    Wstar=Mutation(Wstar,nonelite,agents,probm,variable_list,fixed_rows,variable_rows,sum_fixed,sigma)
    #Impliment selection operator
    Wstar=Survival(W,Wstar,time,X,chromosomes,nonelite,agents,model,lambda,bounded,delta,power,min,max)
    #Replace parent chromosomes with offspring
    W[1:nonelite]=Wstar
    #Store deviation of elite chromosome from current iteration
    old=deepcopy(new)
  end
  return W[chromosomes]
end
