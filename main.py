from pyts import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import shap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tools import catch22_feature_extraction, word_feature_vectors_extraction, matrix_corr

dataset_lst = datasets.ucr_dataset_list()

data = datasets.fetch_ucr_dataset("Trace", use_cache=True, data_home=None, return_X_y=False)
X_train = data["data_train"]
y_train = data["target_train"]
X_test = data["data_test"]
y_test = data["target_test"]

X_train = X_train[:100, :]

#1 - feature extraction
X_catch22 = catch22_feature_extraction(X_train)
X_catch22_test = catch22_feature_extraction(X_test)
#2 - word feature extraction
X_words = word_feature_vectors_extraction(X_train)

# plt.figure()
# sns.heatmap(X_catch22[0])
# plt.figure()
# sns.heatmap(X_words[0])
# plt.show()

#3 - correlate words with features
X_corr = matrix_corr(X_catch22[0], X_words[0])
pd_X_corr = pd.DataFrame(X_corr, index=X_catch22[1], columns=X_words[1])
# sns.heatmap(pd_X_corr, cmap="viridis")
# plt.show()

#4 - Classify
clf = RandomForestClassifier(max_depth=10).fit(X_catch22[0], y_train)
y_pred_test = clf.predict(X_catch22_test[0])
print(confusion_matrix(y_true=y_test, y_pred=y_pred_test))

#5 - Shap mapping
# df_X_catch22_train = pd.DataFrame(X_catch22[0], columns=X_catch22[1])
# explanation = shap.TreeExplainer(clf, df_X_catch22_train, model_output="probability")
# shap_values = explanation.shap_values(df_X_catch22_train)
# plt.figure()
# ax1 = plt.subplot(122)
# sns.heatmap(pd_X_corr.abs(), cmap="viridis", ax=ax1)
# plt.subplot(121)
# shap.summary_plot(shap_values[0], df_X_catch22_train)
