import numpy as np
import h5py

from sklearn.svm import SVC


# load some training data
f = h5py.File('celebfeatures.hdf5','r')
X = f['features'][0:2000]
y = f['sex'][0:2000]

# regress using a SVM
clf = SVC(C=0.1, cache_size=500, kernel='sigmoid')

clf.fit(X, y)
print "Classification done"


# This is the obtained classifier
c0 = clf.intercept_.copy()
alpha = clf.dual_coef_.copy()
vectors = clf.support_vectors_.copy()
gamma = 1./X.shape[1]
# this is only for test
dec = lambda x : np.sign(c0 + (alpha* np.tanh(gamma * vectors.dot(x))).sum())
# serialisation
with open("svm.txt", "w") as out:
    out.write("{:d} {:d}\r\n".format(vectors.shape[0], vectors.shape[1]))
    out.write("{:f}\r\n".format(gamma))
    out.write("{:f}\r\n".format(c0[0]))

    for i in range(alpha.shape[1]):
        out.write("{:f} ".format(alpha[0, i]))
    out.write("\r\t")

    for i in range(vectors.shape[0]):
        for j in range(vectors.shape[1]):
            out.write("{:f} ".format(vectors[i, j]))
        out.write("\r\t")
    out.flush()
    out.close()

# test on other data chunks (separated for speed ...)
for i in range(3):
    Xv = f['features'][10000* (i+1):10000* (i+2)]
    Yv = f['sex'][10000* (i+1):10000* (i+2)]
    print "Classification rate :",
    print float(np.equal(clf.predict(Xv), Yv).sum()) / Yv.shape[0]

f.close()
