from scanf import scanf

# Computes minimum indices of 2d array in Fortran namelist
def min_max_indices_2d(varName,inputFilename):
  varName = varName.lower()
  index_1 = []
  index_2 = []
  with open(inputFilename, 'r') as f:
    inputFile = f.readlines()
    for line in inputFile:
        line3 = line.strip().lower()
        find_index = line3.find(varName+'(')
        # Line contains desired varName
        if (find_index > -1):
          out = scanf(varName+"(%d,%d)",line[find_index::].lower())
          index_1.append(out[0])
          index_2.append(out[1])
  return min(index_1), min(index_2), max(index_1), max(index_2)

def namelistLineContains(line,varName):
  line2 = line.strip().lower()
  varName = varName.lower()
  # We need enough characters for the varName, =, and value:
  if len(line2)<len(varName)+2:
    return False
  if line2[0]=="!":
    return False
  nextChar = line2[len(varName)]
  if line2[:len(varName)]==varName and (nextChar==" " or nextChar=="=" or nextChar=="("):
    return True
  else:
    return False