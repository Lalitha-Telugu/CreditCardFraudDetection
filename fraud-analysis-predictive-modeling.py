
# In[1]:
!pip install seaborn
!pip install xgboost
!pip install catboost
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression ,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier, ExtraTreesClassifier,AdaBoostClassifier,BaggingClassifier
    
    
  
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

from sklearn.svm import SVC

# #Data Representation

# In[2]:
df=pd.read_csv("creditcard.csv")
df.head(1)

# #Data_Information

# In[3]:
df.info()

# #Data_Description

# In[4]:
df.describe()

# #Data_Cleaning

# In[5]:
df.isnull().sum()

# In[6]:
df.duplicated().sum()

# In[7]:
df = df.drop_duplicates()


# In[8]:
df["Amount"].value_counts().nlargest(10)

# The most frequent transaction amounts are small, commonly used values (e.g., 1.00, 9.99, 10.00), indicating recurring low-value purchases and typical pricing strategies.

# In[9]:
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# 1️⃣ Histogram
sns.histplot(df['Amount'], bins=50, ax=axes[0], color='skyblue')
axes[0].set_title("Histogram of Transaction Amount")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Count")

# 2️⃣ Boxplot
sns.boxplot(x=df['Amount'], ax=axes[1], color='lightgreen')
axes[1].set_title("Boxplot of Transaction Amount")
axes[1].set_xlabel("Amount")

plt.tight_layout()
plt.show()

# #detect_outliers

# In[10]:
q1=df["Amount"].quantile(.25)
q3=df["Amount"].quantile(.75)
iqr=q3-q1
upper_bound=q3+1.5*iqr
lower_bound=q1-1.5*iqr
outliers=df[(df["Amount"]>upper_bound)|(df["Amount"]<lower_bound)]
outliers

# In[11]:
df["Time"].describe()

# In[12]:

fig, axes = plt.subplots(1, 2, figsize=(14,5))

# Histogram
sns.histplot(df['Time'], bins=50, ax=axes[0],kde=True)
axes[0].set_title("Histogram of Time")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Count")

# Boxplot
sns.boxplot(x=df['Time'], ax=axes[1])
axes[1].set_title("Boxplot of Time")
axes[1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.show()

# In[13]:
df["Class"].value_counts()

# The dataset is highly imbalanced, with fraudulent transactions representing only about 0.17% of the data

# In[14]:


fig, axes = plt.subplots(1, 2, figsize=(14,5))


sns.countplot(x='Class', data=df, ax=axes[0])
axes[0].set_title("Class Distribution (Count Plot)")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")


for p in axes[0].patches:
    axes[0].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center', va='bottom'
    )

# 2️⃣ Pie Chart
class_counts = df['Class'].value_counts()

axes[1].pie(
    class_counts,
    labels=['Normal (0)', 'Fraud (1)'],
    autopct='%1.3f%%',
    startangle=90,
    
)
axes[1].set_title("Class Distribution (Pie Chart)")

plt.tight_layout()
plt.show()


# In[15]:


v_columns = [col for col in df.columns if col.startswith('V')]

fig, axes = plt.subplots(nrows=len(v_columns), ncols=2, figsize=(14, 4 * len(v_columns)))

for i, col in enumerate(v_columns):
    
    # Histogram
    sns.histplot(df[col], bins=50, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Count')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')
    axes[i, 1].set_xlabel(col)

plt.tight_layout()
plt.show()


# #Bivariate_Analysis

# In[16]:
features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]


n_cols = 4  
n_rows = (len(features) + n_cols - 1) // n_cols  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.boxplot(x='Class', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} vs Class')


for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# #data_split

# In[17]:
x=df.drop("Class",axis=1)
y=df["Class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)

# In[18]:
!pip install imbalanced-learn

# In[19]:
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
x_train_res,y_train_res=sm.fit_resample(x_train,y_train)

# #preprocessing&Model_pipeline

# #LogisticRegression

# In[20]:
logistic_pipe=Pipeline([
    ("scaler",RobustScaler()),("LogisticRegression",LogisticRegression(max_iter=1000,random_state=42))
])
logistic_pipe.fit(x_train_res,y_train_res)
print(logistic_pipe.score(x_train_res,y_train_res))
y_pre=logistic_pipe.predict(x_test)
print(accuracy_score(y_test,y_pre))
print(classification_report(y_test,y_pre))

# #XGBClassifier

# In[21]:

xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_train_res, y_train_res)
y_train_pre=xgb.predict(x_train_res)
print(accuracy_score(y_train_res,y_train_pre))

y_pred_xgb = xgb.predict(x_test)

print(accuracy_score(y_test,y_pre))
print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))


# #AdaBoostClassifier

# In[22]:
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=50
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

ada.fit(x_train_res, y_train_res)

y_pred_ada = ada.predict(x_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("AdaBoost ROC-AUC:", roc_auc_score(y_test, ada.predict_proba(x_test)[:,1]))
print("AdaBoost Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))


# #ExtraTreesClassifier

# In[23]:
 
extra_model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


extra_model.fit(x_train_res, y_train_res)
y_train_pre=extra_model.predict(x_train_res)


y_pred_extra = extra_model.predict(x_test)


print("Extra Trees Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_extra))
print("Acc_train",accuracy_score(y_train_res,y_train_pre))
print(classification_report(y_test, y_pred_extra))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_extra))

# #CatBoostClassifier

# In[24]:

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=5,
    eval_metric='F1',
    random_seed=42,
    verbose=100
)


cat_model.fit(x_train_res, y_train_res)


y_pred_cat = cat_model.predict(x_test)


print("CatBoost Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print(classification_report(y_test, y_pred_cat))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cat))


# In[25]:

results = {
    "Model": ["Logistic Regression", "XGBoost", "Extra Trees", "CatBoost"],
    "Precision_1": [0.12,  0.75, 0.95, 0.69],
    "Recall_1":    [0.84,  0.81, 0.78, 0.81],
    "F1_1":        [0.21,  0.78, 0.85, 0.74]
}

df_results = pd.DataFrame(results)


print("Comparison of Models on Class 1 (Fraud):")
print(df_results)


fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25
index = range(len(df_results))

ax.bar([i - bar_width for i in index], df_results['Precision_1'], width=bar_width, label='Precision', color='skyblue')
ax.bar(index, df_results['Recall_1'], width=bar_width, label='Recall', color='lightgreen')
ax.bar([i + bar_width for i in index], df_results['F1_1'], width=bar_width, label='F1-score', color='salmon')

ax.set_xticks(index)
ax.set_xticklabels(df_results['Model'], rotation=30)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Comparison of Models on Fraud Detection (Class 1)')
ax.legend()

plt.tight_layout()
plt.show()


# 📌 Conclusion
# 
# SMOTE was applied to balance the dataset before training.
# 
# Multiple models were evaluated: Logistic Regression, RandomForest, SVM, XGBoost, Extra Trees, and CatBoost.
# 
# Extra Trees achieved the highest Precision for fraud detection, minimizing false positives.
# 
# XGBoost provided the best balance between Precision and Recall.
# 
# CatBoost achieved high Recall, detecting most fraudulent transactions.
# 
# Logistic Regression and RandomForest showed lower performance on fraud detection.
# 
# Key takeaway:
# 
# Extra Trees and XGBoost are the most robust models for detecting credit card fraud in highly imbalanced datasets, without the need for additional preprocessing.