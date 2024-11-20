import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def Preprocessing(path):
    df=pd.read_csv(path)
    df= df.dropna(axis=1, how='all')
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Calculate the mean value of the restecg column

    for i in df.columns():
        mean_col=df[i].mean()
        df[i].fillna(mean_col,inplace=True)
        
    # Step 1: Automatically detect binary and multiclass columns
    binary_columns = []
    multiclass_columns = []
    for col in df.columns:
        unique_vals = df[col].nunique()
        
        if unique_vals == 2:
            binary_columns.append(col)
        elif unique_vals > 2 and df[col].dtype == 'object':  # Categorical but not numeric
            multiclass_columns.append(col)

    # Step 2: Binary quantization (0 or 1)
    for col in binary_columns:
        # Assuming binary values are either True/False or 0/1 or similar
        df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})

    # Step 3: Multiclass quantization
    label_encoders = {}
    for col in multiclass_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Step 4: Normalize the encoded columns (optional)
    scaler = MinMaxScaler()
    columns_to_scale = binary_columns + multiclass_columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Save the processed dataset (optional)
    df.to_csv('processed_dataset.csv', index=False)

    # Output the result
    print("Binary columns:", binary_columns)
    print("Multiclass columns:", multiclass_columns)
    print("Processed dataset:")
    print(df.head())
    X_train, X_test=train_test_split(df,test_size=0.2,random_state=42)
    norm=Normalizer()
    X_train_norm=norm.fit_transform(X_train)
    X_test_norm=norm.transform(X_test)
    heart_disease_df_X = df.drop(columns=['target'])
    heart_disease_df_y = df['target']
    X_train, X_test, y_train, y_test=train_test_split(heart_disease_df_X,heart_disease_df_y,test_size=0.2,random_state=42)