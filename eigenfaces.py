from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import operator
from numpy import linalg as LA

def step4(X_train1):
    print("\n\nmax min image for PC1\n")
    X_train1.min()
    index=np.argmin(X_train1, axis=0)
    print(index)
    rs=X_train[index].reshape(87,65)
    plt.imshow(rs,cmap='gray')
    plt.show()
    index=np.argmax(X_train1, axis=0)
    rs=X_train[index].reshape(87,65)
    plt.imshow(rs,cmap='gray')
    plt.show()

def step7n8(c_train):
    pm=[]
    info=1
    total=1
    w, v = LA.eig(c_train)
    idx = w.argsort()[::-1]

    w = w[idx]
    v = v[:, idx]
    total=np.sum(w)
    k=0
    
    while((info/total)< 0.95):
        info+=w[k]
        k+=1
    print("Number of principle components needed to represent 95% of information:",k)    
    for i in range(k):
        pm.append(v[:,i])
    usethis=np.array(pm)

    evec=np.array(usethis).transpose()
    proj=np.dot(X_train[0,:],evec)
        #print(proj.shape)
        #print(proj)
        #proj=np.dot(proj,evec)
    np.array(proj)
    y = np.expand_dims(proj, axis=0)
        #print(y.shape)
    finalproj=np.dot(evec,proj)
        #print(finalproj.shape)
    rs=finalproj.reshape(87,65)
    plt.imshow(rs,cmap='gray')
    plt.show()    

def step5n6(evec,X_train):
    rs=evec.reshape(87,65)
    #print(rs.shape)
    plt.imshow(rs,cmap='gray')
    plt.show()
    #print(X_train[0,:].shape)
    proj=np.dot(X_train[0,:],evec)
    #print(proj.shape)
    #print(proj)
    #proj=np.dot(proj,evec)
    np.array(proj)
    y = np.expand_dims(proj, axis=0)
    #print(y.shape)
    finalproj=np.dot(evec,proj)
    #print(finalproj.shape)
    rs=finalproj.reshape(87,65)
    plt.imshow(rs,cmap='gray')
    plt.show()
    

def plotgraph(finalModel):
    plt.scatter(finalModel[:,0], finalModel[:,1],c="green", s=3, )
    plt.show()

def eigenV(data,cov,k):
    pm=[]
    w, v = LA.eig(cov)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    for i in range(k):
        pm.append(v[:,i])
    usethis=np.array(pm)
    
    return(usethis.transpose(),w)




def covar(X):
    xt=X.transpose()
    xtx=np.dot(xt,X)
    N=len(X)
    xtx=xtx/(N-1)
    return(xtx)
def standardise(X):
    mean= np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    trainxStd = (X-mean)/std
    return trainxStd  
    
    
    
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape=people.images[0].shape
# fig, axes= plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
# for target, image , ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image, cmap=cm.gray)
#     ax.set_title(people.target_names[target])
# print("people.images.shape: {}".format(people.images.shape))
# print("Number of classes: {}".format(len(people.target_names)))
counts=np.bincount(people.target)
for i,(count,name)in enumerate(zip(counts,people.target_names)) :
    print("{0:25}{1:3}".format(name,count),end=' ')
    if(i+1)%3==0:
        print()
mask =np.zeros(people.target.shape,dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]]=1
X_people=people.data[mask]
y_people=people.target[mask]
X_people=X_people/255.        
X_train,X_test,y_train,y_test=train_test_split(X_people,y_people,stratify=y_people,random_state=0)
# knn=KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train,y_train)
# print("Test set score of 1−nn: {:.2f}".format(knn.score(X_test,y_test)))
# knn=knn.score(X_test,y_test)
# myknn=calcknn(X_train, X_test, y_train, y_test)
# print("Score of My version of KNN",myknn)
# checkAccuracy(knn,myknn)
#pca reduction to 100D
X_train=np.array(X_train)
X_train=standardise(X_train)
c_train=covar(X_train)
k=1
evec2,evtrain2=eigenV(X_train,c_train,2)

X_train2 = np.matmul(X_train, evec2)
evec,evtrain=eigenV(X_train,c_train,1)

X_train1 = np.matmul(X_train, evec)
step4(X_train1)


print("shape afterwards", X_train1.shape)
# X_test=np.array(X_test)
# X_test=standardise(X_test)
# c_test=covar(X_test)
# k=1

# evec1,evtest=eigenV(X_test,c_test,100)
# X_test = np.dot(X_test, evec)


# myknn=calcknn(X_train, X_test, y_train, y_test)
# print("100D KNN",myknn)

# X_train=whitening(X_train,evtrain)
# print("shape afterwards w", X_train.shape)
# X_test=whitening(X_test,evtest)    

# myknn=calcknn(X_train, X_test, y_train, y_test)
# print("100D KNN after whitening",myknn)

step5n6(evec,X_train)
step7n8(c_train)
