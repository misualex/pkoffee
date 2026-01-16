from pathlib import Path
import os

test_data_path = Path(os.getenv("PKOFFEE_DATA", "analysis"))

def test_initial():
    from pkoffee.data import load_csv,pd

    assert 1 == 1
    #assert fibonacci(1) == 1

def test_load_csv():
    from pkoffee.data import load_csv,pd

    # Get the path relative to this test file
    csv_path = test_data_path / "coffee_productivity.csv"
    
    df = load_csv(csv_path)
    assert isinstance(df, pd.DataFrame)

def test_column_names():
    from pkoffee.data import load_csv,pd
    # Get the path relative to this test file
    csv_path = test_data_path / "coffee_productivity.csv"
    
    df = load_csv(csv_path)
    assert 'cups' in df.columns
    assert 'productivity' in df.columns