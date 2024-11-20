from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.svm import SVC
from data_preprocessing import Preprocessing
df=Preprocessing('./heart_disease_df_1.csv')
heart_disease_df_X = df.drop(columns=['target'])
heart_disease_df_y = df['target']
X_train, X_test, y_train, y_test=train_test_split(heart_disease_df_X,heart_disease_df_y,test_size=0.2,random_state=42)
norm=Normalizer()
X_train_norm=norm.fit_transform(X_train)
X_test_norm=norm.transform(X_test)
y_train_norm=norm.transform(y_train)
y_test_norm=norm.transform(y_test)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
rf.fit(X_train_norm, X_test_norm)

# Define the feature selection object
model = SelectFromModel(rf, prefit=True)

# Transform the training features
X_train_transformed = model.transform(X_train)

original_features = df.columns[:-1]
print(f"Original features: {original_features}")

# Select the features deemed important by the SelectFromModel
features_bool = model.get_support()

selected_features = original_features[features_bool]
print(f"\nSelected features: {selected_features}")

feature_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": rf.feature_importances_[features_bool]
})
svc_model = SVC(kernel='linear')
svc_model.fit(X_train_norm, y_train_norm)

kf = KFold(n_splits=5)
# Compute the cross-validation score
score = cross_val_score(svc_model, heart_disease_df_X, heart_disease_df_y, scoring="balanced_accuracy", cv=kf)
print(score)


