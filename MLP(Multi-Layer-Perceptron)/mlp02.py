from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
Y = [0, 1]

# solver = Decision of Algorithm using for Optimization
# lbgfs = Example of Limited-memory Quasi-Newton methods
# 'lbgfs' usually using Hessain matrix calculating or not reasonable cost
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

clf.fit(X, Y)
print(clf.fit)

# predict = predict result of input data
clf.predict([[2., 2.], [-1., -2.]])
print(clf.predict)

[coef.shape for coef in clf.coefs_]
print([coef.shape for coef in clf.coefs_])

clf.predict_proba([[2., 2.], [1., 2.]])
print(clf.predict_proba)