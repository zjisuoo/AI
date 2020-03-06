from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
Y = [[0, 1], [1, 1]]

# solver = Decision of Algorithm using for Optimization
# lbgfs = Example of Limited-memory Quasi-Newton methods
# 'lbgfs' usually using Hessain matrix calculating or not reasonable cost
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

clf.fit(X, Y)
print(clf.fit)

clf.predict([[1., 2.]])
print(clf.predict)

clf.predict([[0., 0.]])
print(clf.predict)