import unittest
import mock
import netCDF4 as cdf
from meld import vault


class TestVaultInit(unittest.TestCase):
    "Test if vault class handles initialization correctly"

    def tearDown(self):
        vault._drop()

    def test_open_for_append_calls_cdf_correctly(self):
        "open_for_append should call cdf.Dataset correctly"
        with mock.patch('meld.vault.cdf', spec=cdf) as mock_cdf:
            dataset = vault.open_dataset_for_append()

            mock_cdf.Dataset.assert_called_once_with('results.progress', mode='a')
            self.assertEqual(dataset, mock_cdf.Dataset.return_value)

    def test_calling_open_for_append_twice_raises(self):
        "calling open_for_append twice should raise RuntimeError"
        with mock.patch('meld.vault.cdf', spec=cdf):
            with self.assertRaises(RuntimeError):
                vault.open_dataset_for_append()
                vault.open_dataset_for_append()

    def test_open_for_read_calls_cdf_correctly(self):
        "open_for_read should call cdf.Dataset correctly"
        with mock.patch('meld.vault.cdf', spec=cdf) as mock_cdf:
            dataset = vault.open_dataset_for_read()

            mock_cdf.Dataset.assert_called_once_with('results.nc', mode='r')
            self.assertEqual(dataset, mock_cdf.Dataset.return_value)

    def test_calling_open_for_read_twice_raises(self):
        "calling open_for_read twice should raise RuntimeError"
        with mock.patch('meld.vault.cdf', spec=cdf):
            with self.assertRaises(RuntimeError):
                vault.open_dataset_for_read()
                vault.open_dataset_for_read()

    def test_open_for_write_calls_cdf_correctly(self):
        "open_for_write should call cdf.Dataset correctly"
        with mock.patch('meld.vault.cdf', spec=cdf) as mock_cdf:
            dataset = vault.open_dataset_for_write()

            mock_cdf.Dataset.assert_called_once_with('results.progress', mode='w', clobber=False)
            self.assertEqual(dataset, mock_cdf.Dataset.return_value)

    def test_calling_open_for_write(self):
        "calling open_for_write twice should raise RuntimeError"
        with mock.patch('meld.vault.cdf', spec=cdf):
            with self.assertRaises(RuntimeError):
                vault.open_dataset_for_write()
                vault.open_dataset_for_write()

    def test_transfer_progress(self):
        "transfer progress should sync and call copy"
        with mock.patch('meld.vault.cdf', spec=cdf), mock.patch('meld.vault.shutil.copy') as mock_copy:
            dataset = vault.open_dataset_for_write()
            vault.transfer_dataset_progress()

            dataset.sync.assert_called_once_with()
            mock_copy.assert_called_once_with('results.progress', 'results.nc')
