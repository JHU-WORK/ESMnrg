# import
import os
import pandas as pd
import pytest
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock

# Custom imports
from data.utils import get_loader, create_csv_from_subset

def test_create_csv_from_subset(tmpdir):
    # Create a temporary CSV file to act as input
    dataset_path = tmpdir.join("input.csv")
    output_path = str(tmpdir)
    output_name = "output"

    # Sample data
    data = {'name': ['Alice', 'Bob'],
            'dna_seq': ['ATCG', 'CGTA'],
            'deltaG': [1.0, 2.0],
            'extra_column': [5, 6]}
    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)

    # Columns to be extracted
    columns = ['name', 'dna_seq', 'deltaG']

    # Call the function under test
    create_csv_from_subset(str(dataset_path), columns, output_path, output_name)

    # Read the output file and verify
    output_file = os.path.join(output_path, f"{output_name}.csv")
    result_df = pd.read_csv(output_file)
    expected_df = df[columns]
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_get_loader():
    # Setup code: Create a mock CSV file for DataLoader
    test_data = pd.DataFrame({
        'aa_seq': ['VEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER', 'IEVTIHLGDKTIRVDGLDKELLEILKELARRGADEEELRKEIERWER'],
        'deltaG': [3.22093297, 3.27675637]
    })
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)

    # Patch ESMnrgDataset to return a simple dataset
    with patch('data.datasets.ESMnrgDataset') as mock_dataset_class:
        # Create a mock dataset instance
        mock_dataset_instance = MagicMock()
        mock_dataset_class.return_value = mock_dataset_instance
        
        # Configure the return value for the dataset loader (mocking behavior)
        mock_dataloader = DataLoader(mock_dataset_instance, batch_size=2, shuffle=True, num_workers=1)
        
        # Call the function under test
        loader = get_loader(file_path=test_path, file_type='csv', batch_size=2, shuffle=True, num_workers=1, dataset='train')
        
        # Assert that ESMnrgDataset is initialized correctly
        mock_dataset_class.assert_called_once_with(file_path=test_path, file_type='csv', partition='train',
                                                   train_split=0.8, val_split=0.1, test_split=0.1, seq_len=240)
        
        # Assertions to check if DataLoader is setup correctly
        assert loader.batch_size == 2
        assert loader.shuffle == True
        assert loader.num_workers == 1

    # Cleanup code
    os.remove(test_path)
