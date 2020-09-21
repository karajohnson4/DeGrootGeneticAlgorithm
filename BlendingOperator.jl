function Blending(W,nonelite,agents,probb)
  #W:
  #chromosomes: number of chromosomes
  #nonelite: number of non-elite chromosomes (chromosomes-1)
  #agents: number of agents
  #probb: probability of blending
  #Performes the blending operator
  #Determine random reordering for pairing
  reorder=randperm(nonelite)
  #Populate offspring chromosomes with current parent chromosomes excluding elite
  Wstar=W[1:nonelite]
  #Loop through all pairs of chromosomes
  for k in 1:2:(nonelite-1)
    #Loop through all agents (rows)
    for i in 1:agents
      #Probability check for blending row i
      if (rand()<=probb)
        #Generate blending factor
        beta=rand()
        #Create new row for first chromosome in pair
        Wstar[reorder[k]][i,:]=beta.*W[reorder[k]][i,:]+(1-beta).*W[reorder[k+1]][i,:]
        #Create new row for second chromosome in pair
        Wstar[reorder[k+1]][i,:]=(1-beta).*W[reorder[k]][i,:]+beta.*W[reorder[k+1]][i,:]
      end
    end
  end
  return Wstar
end
