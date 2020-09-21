function Scale(response::Array,min::Int,max::Int,back::Bool=false)
  #Converts likert-scale data to continuous (0,1) and back
  #response: N x M matrix of either likert-scale or continuous scale responses
  #min: minimum value on likert scale
  #max: maximum value on likert scale
  #back: optional logical, if true responses are back-transformed to a likert-scale

  response1=deepcopy(response)

  #Check max excedes min
  if max<=min
    error("Scale: max ($max) must excede min ($min)")
  end
  #Count total number of responses
  responses=length(response1)
  #Count number of possible responses to determine scaling factor
  scale=length(min:max)
  #Determine shift for likert scale to start at 1
  internal=1-min
  #Forward transformation
  if back==false
    #Loop through all responses
    for i in 1:responses
      #Check all values are either missing or integers between min and max
      if ismissing(response1[i])==false && (isinteger(response1[i])==false || response1[i]<min || response1[i]>max)
          error("Scale: elements in response must be either missing or an integer between specified min and max of $min and $max for forward-transformation")
      end
    end
    #Determine shift for after scaling responses
    shift=1/(2*scale)
    #Shift likert scale responses to start at 1
    response1.+=internal
    #Apply scale to responses
    response1=response1./scale
    #Apply shift to response
    response1.-=shift
    #Return scaled responses
    return response1
  #Backward transformation
  else
    #Loop through all responses
    for i in 1:responses
      #Check all values are either missing or between 0 and 1
      if ismissing(response1[i])==false && (response1[i]<0 || response1[i]>1)
        error("Scale: elements in response must either be missing or between 0 and 1 for back-transformation")
      end
    end
    #Rescale response to range of likert-scale values
    response1=response1.*scale.+1/2
    #Round response to integer values
    response1=round.(response1)
    #Shift responses to original likert scale
    response1.-=internal
    #Return scaled responses
    return response1
  end
end
