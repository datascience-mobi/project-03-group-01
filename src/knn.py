distancevector=[]
testvector=[]
testvectordigit=-1 # true test image digit
numsuccess=0
totaltries=0 # number of analyzed test images
k=0

# call the relevant functions for performing KNN
def perform_knn(kvalue, testpath, testnumber, trainingpath):
    # set k as global variable
    set_k(kvalue)
    # assign intensity values from test image to global variable
    get_test_vector(testpath, testnumber)
    # loop through all training images to perform KNN
    get_training_vectors(trainingpath)

# after performing KNN for one test image reset the global variables
def reset_global_vars():
    global distancevector, testvector, testvectordigit, k
    distancevector = []
    testvector = []
    testvectordigit = -1
    k = 0

# loop through training vectors to create an array containing distance and digit for each training image
def get_training_vectors(path):
    global distancevector, k
    distancevector.append([]) # TODO: maybe useless?

    with open(path) as infile:
        for line in infile:
            trainvector=line.split(",") # fill trainvector with intensity values from one line/image, see csv structure
            distance=get_distance(trainvector,testvector) # distance between test vector and current training vector
            if not distance == -1:
                distancevector.append([distance,int(trainvector[0])])
            else:
                return(-1) # check for errors in get_distance function
        sorteddistancevector=distancevector # new array to keep unsorted one, maybe redundant
        sorteddistancevector.pop(0) # remove first element (label)
        sorteddistancevector.sort() # ascending order -> first k elements have lowest distance
        # create array of nearest digits
        results=[]
        for i in range(k):
            #print(distancevector[i]) # used for debugging
            #print(len(distancevector[i])) # used for debugging
            results.append(distancevector[i][1]) # append second element of distancevector: digit of original vector

        process_results(results)

        reset_global_vars()

# find most frequent nearest neighbors and get success rate e.g. for plotting it against k
def process_results(results):
    global numsuccess, totaltries, testvectordigit
    print("True Result= " + str(testvectordigit))
    nummaxmatches = 0  # TODO: figure out variable naming convention
    digitmaxmatches = results[0]
    for i in results:
        freq = results.count(i)
        if freq > nummaxmatches:
            nummaxmatches = freq
            digitmaxmatches = i
    print("Result= " + str(digitmaxmatches) + " with counts: " + str(nummaxmatches))
    totaltries+=1
    print("Expected result: "+str(testvectordigit)+" - result: "+str(digitmaxmatches)+" - matching? "+str(digitmaxmatches==testvectordigit))
    if digitmaxmatches==testvectordigit:
        numsuccess+=1
        print("SUCCESS")
    if not totaltries==0:
        successrate=float(numsuccess)/float(totaltries)*float(100)
    else:
        successrate=0
    print("Current success rate: "+str(successrate)+"%")


# set value for k variable
def set_k(kvalue):
    global k
    k=kvalue

# calculates euclidian distance between two vectors (test image and one training image)
def get_distance(vector1, vector2):
    if not len(vector1) == len(vector2): # check if vectors have same dimensionality
        print("ERROR, incompatible image sizes: "+str(len(vector1))+" and "+str(len(vector2)))
        return(-1) # abort, provide information for further analysis
    sum=0
    first=True # TODO: remove first element (label) in the beginning without skipping here
    for i in range(len(vector1)): # for each element calculate squared distance
        a=int(vector1[i])-int(vector2[i])
        if not first==True:
            sum+=a*a
        first=False
    return(sum)

# linenumber for a specific test image was passed as parameter -> find and process it
def get_test_vector(path, linenumber):
    global testvector, testvectordigit
    with open(path) as infile:
        number=0 # manual counter for finding the line, maybe obsolete
        for line in infile:
            number+=1
            if number == linenumber:
                testvector = line.split(",") # elements between each , are one element of testvector
                #testvector[len(testvector)-1]=testvector[len(testvector)-1].strip() # probably redundant
                testvectordigit=int(testvector[0])
                #print(testvector) # lines are strings TODO: convert to string and remove \n from last element
                return # break out of the loop once the line is found and processed
