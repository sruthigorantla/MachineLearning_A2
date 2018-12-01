# Mathieu Blondel, September 2010
# License: BSD 3 clause

import numpy as np
from numpy import linalg
import cvxopt
import matplotlib.pyplot as plt
import cvxopt.solvers
import sys
import ans1a as a1a
import ans1d as a1d
import pickle

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print(X.shape)
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])


        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), tc='d')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        print(np.linalg.matrix_rank(P))
        print(np.linalg.matrix_rank(q))
        print(np.linalg.matrix_rank(A))
        print(np.linalg.matrix_rank(b))
        print(np.linalg.matrix_rank(G))
        print(np.linalg.matrix_rank(h))
        # sys.exit(0)
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            print(type(ind[n]),type(sv[0]))
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        print(self.a,self.b)
        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            print("length of a: ",len(self.a))
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl

    # def gen_lin_separable_data():
    #     # generate training data in the 2-d case
    #     mean1 = np.array([0, 2])
    #     mean2 = np.array([2, 0])
    #     cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    #     X1 = np.random.multivariate_normal(mean1, cov, 100)
    #     y1 = np.ones(len(X1))
    #     X2 = np.random.multivariate_normal(mean2, cov, 100)
    #     y2 = np.ones(len(X2)) * -1
    #     return X1, y1, X2, y2
    def dummy_fn(x):
        if np.sum(x) == 0:
            return -1
        return np.sign(np.sum(x))

    def gen_lin_separable_data(n=100, f=dummy_fn, lims=[5, 5]):
        # d = len(lims)
        # print(n)
        # # Intialize the data
        # X = np.zeros((n, d))
        # Y = np.zeros((n, 1))
        Y1 = []
        Y2 = []
        X1 = []
        X2 = []

        with open("data.pickle","rb") as fp:
            (X,Y) = pickle.load(fp)

        for i in range(len(X)):
            # X[i][0] = np.random.uniform(-lims[0],lims[0])
            # X[i][1] = np.random.uniform(-lims[1],lims[1])
            # Y[i] = f(X[i])
            if(Y[i] == 1):
                Y1.append(1)
                X1.append(X[i])
            else:
                Y2.append(-1)
                X2.append(X[i])
        # Obtaining the label y based on f
        plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y, s=5)
        
        plt.show()
        Y1 = np.asarray(Y1)
        Y2 = np.asarray(Y2)
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        return X1,Y1,X2,Y2

    def gen_non_lin_separable_data():
        X, Y = a1a.data_gen(n=1000, f=a1d.complex_fn)
        Y1 = []
        Y2 = []
        X1 = []
        X2 = []
        
        for i in range(len(X)):
            # X[i][0] = np.random.uniform(-lims[0],lims[0])
            # X[i][1] = np.random.uniform(-lims[1],lims[1])
            # Y[i] = f(X[i])
            if(Y[i] == 1):
                Y1.append(1)
                X1.append(X[i])
            else:
                Y2.append(-1)
                X2.append(X[i])
        # Obtaining the label y based on f
        plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y, s=5)
        
        plt.show()
        Y1 = np.asarray(Y1)
        Y2 = np.asarray(Y2)
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        return X1,Y1,X2,Y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        ind = int(0.8*len(X1))
        X1_train = X1[:ind]
        y1_train = y1[:ind]
        X2_train = X2[:ind]
        y2_train = y2[:ind]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        ind = int(0.8*len(X1))
        X1_test = X1[ind:]
        y1_test = y1[ind:]
        X2_test = X2[ind:]
        y2_test = y2[ind:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data(n=1000)
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=0.1)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    test_non_linear()