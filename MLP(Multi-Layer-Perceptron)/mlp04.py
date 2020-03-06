from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler()

X_train = [[0., 0.], [1., 1.]]
X_test = [[0, 1], [1, 1]]
Y = [0, 1]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# warm_start = after completing a cold start, Skip self-test procedures
clf = MLPClassifier(hidden_layer_sizes = (15), random_state = 1, max_iter = 1, warm_start = True)

for i in range(10):
    clf.fit(X_train, Y)
    print(clf.fit)
    clf.fit(X_test, Y)
    print(clf.fit)