distancevector=[]
testvector=[]
testvectordigit=-1
k=0

def perform_knn(kvalue, testpath, testnumber, trainingpath):
    set_k(kvalue)
    get_test_vector(testpath, testnumber)
    get_training_vectors(trainingpath)

def reset_stuff():
    global distancevector, testvector, testvectordigit, k
    distancevector = []
    testvector = []
    testvectordigit = -1
    k = 0

def get_training_vectors(path):
    global distancevector
    global k
    distancevector.append([])
    with open(path) as infile:
        for line in infile:
            trainvector=line.split(",")
            distance=get_distance(trainvector,testvector)
            if not distance == -1:
                distancevector.append([distance,int(trainvector[0])])
                #distancevector[len(distancevector)-1][1]=trainvector[0]
            else:
                return(-1)
        sorteddistancevector=distancevector
        sorteddistancevector.pop(0)
        sorteddistancevector.sort()
        results=[]
        for i in range(k):
            #print(distancevector[i])
            #print(len(distancevector[i]))
            results.append(distancevector[i][1])
        print("True Result= "+str(testvectordigit))
        max = 0
        res = results[0]
        for i in results:
            freq = results.count(i)
            if freq > max:
                max = freq
                res = i
        print("Result= "+str(res) +" with counts: "+str(max))
        #print(results)
        reset_stuff()

def set_k(kvalue):
    global k
    k=kvalue

def get_distance(vector1, vector2):
    if not len(vector1) == len(vector2):
        print("ERROR, incompatible image sizes: "+str(len(vector1))+" and "+str(len(vector2)))
        return(-1)
    #else:
    #    print("sizes: " + str(len(vector1)) + " and " + str(len(vector2)))
    sum=0
    first=True
    for i in range(len(vector1)):
        a=int(vector1[i])-int(vector2[i])
        if not first==True:
            sum+=a*a
        first=False
    return(sum)

def get_test_vector(path, linenumber):
    global testvector
    global testvectordigit
    with open(path) as infile:
        number=0
        for line in infile:
            number+=1
            if number == linenumber:
                #print(line)
                line.strip()
                #print(line)
                testvector = line.split(",")
                testvector[len(testvector)-1]=testvector[len(testvector)-1].strip()
                testvectordigit=testvector[0]
                #testvector.pop(0)
                #print(testvector)
                return
