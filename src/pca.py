import shutil, os

def perform_pca(training_path):
    """For normalization: Get vactor containing mean values for all pixels and training images"""
    sumintensities=[]
    meanintensities=[]
    numimages=0
    with open(training_path) as infile:
        for line in infile:
            image_vector=line.split(",")
            sumintensities=[0]*len(image_vector)
            meanintensities=[0]*len(image_vector)
    truedigits=[]
    """with open(training_path) as infile:
        for line in infile:
            image_vector=line.split(",")
            truedigits.append(int(image_vector[0]))
            image_vector.pop(0)
            for i in range(len(image_vector)):
                sumintensities[i]+=int(image_vector[i])
            numimages+=1
        for index in range(len(sumintensities)):
            meanintensities[index]=int(round(float(sumintensities[index])/float(numimages)))
        print(meanintensities)
        print("Mean list created")
    shutil.rmtree("../temp", ignore_errors=True)
    os.mkdir("../temp")
    f = open("../temp/normalized_training.csv", "w+")
    with open(training_path) as infile:
        linenumber=0
        for line in infile:
            f.write(str(truedigits[linenumber])+",")
            linenumber+=1
            old_vector=line.split(",")
            for index in range(len(old_vector)):
                f.write(str(int(old_vector[index])-meanintensities[index])+",")
            f.write("\n")
    f.close()
    print("Normalizing completed")"""

    """Create covariance matrix of our X*N Matrix where X=pixelnumber and N=image number
        with the formula of Sigma(i,j)=1/numrows-1 * crossproduct (coli, colj)"""
    covariance_matrix=[]
    templist3=[]
    f=open("../temp/covmatrix_training.csv", "w+")
    for i in range(50,len(sumintensities)):
        templist=collist(i, training_path)
        for j in range(50,len(sumintensities)):
            templist2=collist(j, training_path)
            #print("Sizes: "+str(len(templist))+" and "+str(len(templist2)))
            cov=covariance(len(sumintensities), templist, templist2)
            templist3.append(cov)
            print(cov)
            print(str(i)+" x "+str(j))
        #covariance_matrix.append(templist3)
        print(templist3)
        for t3 in templist3:
            f.write(str(t3)+",")
        f.write("\n")
        print(templist3)
    print("Covariance matrix finished!")
    f.close()

def collist(colnumber, path):
    templist=[]
    with open(path) as infile:
        for line in infile:
            linelist=line.split(",")
            num=0
            for l in range(len(linelist)):
                if num == colnumber:
                    templist.append(int(linelist[l]))
                    break
                num += 1
    #print("Scale: "+str(len(templist))+"   "+str(colnumber))
    return templist

def covariance(n, list1, list2):
    print("n: "+str(n))
    factor1=1.0/float(n-1)
    print(".......÷÷÷÷÷÷÷÷÷ "+str(factor1))
    cov=factor1*cross_product(list1, list2)
    print("-- "+str(cov))
    return cov

def cross_product(list1, list2):
    if not len(list1) == len(list2):
        print("Incompatible types: "+str(len(list1))+" and "+str(len(list2)))
        return -1
    product=0
    for index in range(len(list1)):
        product+=list1[index]*list2[index]
    print("Product: "+str(product))
    return product