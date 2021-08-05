import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

history = model.fit()

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")


model.evaluate(X_test, y_test)


#tf.earlystopping_callback



X["age"].plot(kind="hist") #histogram
X["age"].value_counts()

normalization: make values between 0 and 1, minmax normalization(scaler)
standardization removes teh mean and divides by standard deviation

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

ct = make_column_transformer((MinMaxScaler(),['column','column']),
			     (OneHotEncoder(handle_unknown="ignore"), ['column']))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state=42)

ct.fit(X_train)
X_train-normal = ct.transform(X_train)

#create random data
from sklearn.datasets import make_circles
n_samples = 1000
X, y = make_circles(n_samples, noise = 0.03, random_state = 42)

#visualize a dataset of X1,X2 coordinates and a class (gadwal w plot)
circles = pd.DataFrame({"X0:X[:,0], :"X1":X[:,1], "label":y}
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu);

#visualize the decision boundary
import numpy as np
def plot_decision_boundary(model, X, y):
	x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
	y_min, y_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
			     np.linespace(y_min, y_max, 100))

	x_in = np.c_[xx.ravel(), yy.ravel()]
	y_pred = model.predict(x_in)
	if len(y_pred[0]) > 1:
		y_pred = np.argmax(y_pred,axis = 1).reshape(xx.shape)
	else:
		y_pred = np.round(y_pred).reshape(xx.shape)
	plt.contourf(xx,yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
	plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
	plt.xlim(xx.min(), xx.max())
	plt.ylimt(yy.min(),yy.max())

pd.DataFrame(history.history).plot()

lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch = 1e-4 * 10 **(epoch/20))

#EQ like graph
plt.semilogx()


from sklearn.metrics import confusion_matrix
y_preds = model.predict(X_test)
confusion_matrix(y_test, y_preds)


#prettify a confusion matrix
import itertools

figsize = (10,10)

cm = confusion_matrix(y_test, tf.round(y_preds))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
n_classes = cm.shape[0]

fig, ax = plt.subplots(figsize = figsize)
cax = ax.matshow(cm,cmap=plt.cm.Blues)
fig.colorbar(cax)

classes = False

if classes:
	labels = classes
else:
	labels = np.arange(cm.shape[0])

ax.set(title="Confusion Matrix",
	xlabel="predicted",
	ylabel="true",
	xticks=np.arange(n_classes),
	ytikcs=np.arange(n_classes),
	xticklabels=labels,
	yticklabels=labels)

threshold = (cm.max()+cm.min()) /2.

for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	plt.text(j, i, f"{cm[i, j]}" ({cm_norm[i,j]*100:.1f}%)", horizontalalignment ="center",
	color ="white" if cm[i,j] > threshold else "black", size = 15)


tf.one_hot(list,depth=class_number) #use categorical instead of sparsecategorical



#layers.get_weights()

#plot models
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)
