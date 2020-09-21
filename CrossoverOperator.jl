function Crossover(W,nonelite,agents,probc,variable_list,fixed_rows,variable_rows)
  #Loop through all non-elite chromosomes
  for k in 1:nonelite
    #Loop through all rows without fixed values
    for i in variable_rows
      #Probability check for crossover row i
      if rand()<=probc
        #Randomly reorder all elements in row i
        W[k][i,:]=shuffle(W[k][i,:])
      end
    end
    #Loop through all rows with fixed values
    for i in fixed_rows
      #Probability check for crossover row i
      if rand()<=probc
        #Randomly permute indices of all fixed values
        permutation=shuffle(variable_list[i])
        #Reorder all variable elements in row i using the random permutation
        W[k][i,variable_list[i]]=W[k][i,permutation]
      end
    end
  end
  return W
end
