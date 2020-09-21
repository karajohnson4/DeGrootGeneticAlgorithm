function Mutation(W,nonelite,agents,probm,variable_list,fixed_rows,variable_rows,sum_fixed,sigma)
  #Loop through all non-elite chromosomes
  for k in 1:nonelite
    #Loop through all rows without fixed values
    for i in variable_rows
      #Probability check for mutation row i
      if rand()<=probm
        #Identify index of element to be mutated
        index=rand(1:agents)
        #Randomly generate epsilon for mutation
        epsilon=rand(Normal(0,sigma))
        #Mutate selected value
        value=W[k][i,index]+epsilon
        #Mutated value 1 or greater
        if value>=1
          #Force all values in row i for chromosome k to 0
          W[k][i,:]=zeros((1,agents))
          #Replace mutated value with 1
          W[k][i,index]=1
        #Mutated value less than 1
        else
          #Mutated value less than 0
          if value<0
            #Force mutated value to 0
            value=0
          end
          #Check if weight needs to be distributed manually
          if isapprox(W[k][i,index],1)
            #Determine number of variable elements to distribute weight across
            number=agents-1
            #Distribute extra weight across all variable elements
            W[k][i,variable_list[i]]=fill((1-value)/number,number+1)
            #Overwrite mutated value
            W[k][i,index]=value
          #Weight can be distributed using scaling
          else
            #Find sum of non-mutated values for row i in chromosome k
            sum_row=sum(W[k][i,:])-W[k][i,index]
            #Determine scaling factor so row i of chromosome k sums to 1 after mutation
            scale=sum_row/(1-value)
            #Apply scaling value to row i of chromosome k
            W[k][i,:]/=scale
            #Replace mutated value
            W[k][i,index]=value
          end
        end
      end
    end
    #Loop through all rows with fixed values
    for i in fixed_rows
      #Check if mutation is possible
      if size(variable_list[i])[1]>0
        #Probability check for mutation for row i
        if rand()<=probm
          #Identify index of element to be mutated
          index=rand(variable_list[i])
          #Randomly generate epsilon for mutation
          epsilon=rand(Normal(0,sigma))
          #Mutate selected value
          value=W[k][i,index]+epsilon
          #Check if mutated value is 1 minus the sum of fixed values or greater for row i
          if value>=(1-sum_fixed[i])
            #Force all variable values in row i for chromosome k to 0
            W[k][i,variable_list[i]]=zeros((1,size(variable_list[i])[1]))
            #Replace mutated value with 1 minus the sum of fixed values for row i
            W[k][i,index]=1-sum_fixed[i]
          #Mutated value less than 1
          else
            #Mutated value less than 0
            if value<0
              #Force mutated value to 0
              value=0
            end
            #Find sum of non-mutated and non-fixed values for row i in chromosome k
            sum_row=sum(W[k][i,variable_list[i]])-W[k][i,index][1]
            #Check if sum of non-mutated variable elements is 0 (scaling must be done manually)
            if sum_row==0
              #Determine extra weight to be distributed
              extra=1-value-sum_fixed[i]
              #Determine number of variable elements to distribute weight across
              number=size(variable_list[i])[1]-1
              #Distribute extra weight across all variable elements
              W[k][i,variable_list[i]]=fill(extra/number,number+1)
              #Overwrite mutated value
              W[k][i,index]=value
            #Sum is non-zero
            else
              #Determine scaling factor so row i of chromosome k sums to 1 after mutation
              scale=sum_row/(1-sum_fixed[i]-value)
              #Apply scaling value to row i of chromosome k
              W[k][i,:]/=scale
              #Replace mutated value
              W[k][i,index]=value
            end
          end
        end
      end
    end
  end
  #Bandaid
  #Loop through all non-elite chromosomes
  for k in 1:nonelite
    #Loop through all rows (agents)
    for i in 1:agents
      W[k][i,:]/=sum(W[k][i,:])
    end
  end
  return W
end
