from src.data_loader import load_data
from src.preprocessing import basic_clean

def test_load_and_clean():
    df = load_data()
    assert df.shape[0] > 0,
    df2 = basic_clean(df)
    assert 'Age' in df2.columns
    assert 'TotalSpend' in df2.columns
