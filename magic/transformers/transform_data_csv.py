import pandas as pd

def number_of_rows_per_key(df, key, column_name):
    data = df.groupby(key)[key].agg(['count'])
    data.columns = [column_name]
    return data

def clean_column(column_name):
    return column_name.lower().replace(' ', '_')

def preprocess(df, *args, **kwargs):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print(f"Warning: Missing values found in columns: {print(missing_values[missing_values > 0])}")
    
    # Check for missing value
    required_columns = ['iduser']
    for cols in required_columns:
        if df[cols].isnull().any():
            raise ValueError(f"Missing values detected in critical column: {cols}")

    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates()
        
    df = df.drop(columns=['temp_column'], errors='ignore')

    return df

@transformer
def transform(df, *args, **kwargs):
    # Add number of meals for each user
    df_new_column = number_of_rows_per_key(df, 'Iduser', 'total_row')
    df = df.join(df_new_column, on='Iduser')

    # Clean column names
    df.columns = [clean_column(col) for col in df.columns]
    df = preprocess(df)
    df['start_watching'] = pd.to_datetime(df['start_watching'], format='%m/%d/%Y %H:%M', errors='coerce').dt.tz_localize('UTC')

    return df


@test
def test_number_of_columns(df, *args) -> None:
    assert len(df.columns) < 11, 'There needs to be at least 11 columns.'
