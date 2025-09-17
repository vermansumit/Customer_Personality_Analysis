import pandas as pd
from sklearn.impute import SimpleImputer

def basic_clean(df):
    df = df.copy()
    
    #strip possible BOM from ID column
    if '\ufeffID' in df.columns:
        df = df.rename(columns={'\ufeffID': 'ID'})

    #Age
    from datetime import datetime
    df['Age'] = df['Age'] = datetime.now().year - df['Year_Birth']

    # Total spent
    mnt_cols = [c for c in df.columns if c.startswith('Mnt')]
    df['TotalSpend'] = df[mnt_cols].sum(axis=1)

    #fill income with median
    df['Income'] = df['Income'].replace(' ', pd.NA).astype('float')
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    #simple encode education levels
    df['Education'] = df['Education'].str.strip().str.lower()
    return df