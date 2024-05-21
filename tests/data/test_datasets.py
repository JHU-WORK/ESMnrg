# imports
import os
import pytest
import pandas as pd

# Custom imports
from data.utils import get_loader

# Test the DataLoader class
@pytest.fixture
def test_load_data():
    # Setup: Create a mock CSV file
    test_data = pd.DataFrame({
        'aa_seq': ['VEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER', 'IEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER'],
        'deltaG': [3.22093297, 3.27675637]
    })
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    yield test_path
    # Cleanup code
    os.remove(test_path)

# Optionally, use fixtures to handle setup and cleanup
@pytest.fixture
def test_load_data_with_fixture(create_test_file):
    # Given
    file_path = create_test_file
    batch_size = 2  # Assuming a small batch size for testing

    # When
    loader = get_loader(file_path=file_path, file_type='csv', batch_size=batch_size, shuffle=False, num_workers=0, dataset='train', train_split=1.0, val_split=0.0, test_split=0.0)

    # Then
    for batch in loader:
        # Check the actual loaded data against expected data
        expected_data = pd.DataFrame({
            'aa_seq': ['VEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER', 'IEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER'],
            'deltaG': [3.22093297, 3.27675637]
        })
        pd.testing.assert_frame_equal(pd.DataFrame(batch), expected_data)

# Additional configurations for more detailed pytest outputs
@pytest.fixture
def create_test_file():
    # Setup code: Create a mock CSV file for DataLoader
    test_data = pd.DataFrame({
        'aa_seq': ['VEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER', 'IEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER'],
        'deltaG': [3.22093297, 3.27675637]
    })
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    yield test_path
    # Cleanup code
    os.remove(test_path)
