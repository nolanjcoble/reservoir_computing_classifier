import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier as mlp
import time

start_time = time.time()

input_file = 'seeds_dataset.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)

params = {'random_state': 0, 'hidden_layer_sizes': 50, 'max_iter': 2000}
classifier = mlp(**params)


classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)


print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train)))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred))
print("#"*40 + "\n")

# print running time
print("Running time: " + str(time.time() - start_time) + "sec")
print('# layers: ' + str(classifier.n_layers_))
