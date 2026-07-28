import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RRUtil.DataBase import DB


class DatabaseTest(unittest.TestCase):
    def test_corrupt_database_is_preserved_and_reinitialized(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory, 'analysis.json')
            database_path.write_text('{"MetaData": {}, "Stats": {')

            with warnings.catch_warnings(record=True) as caught:
                database = DB(database_path, 'default')

            self.assertEqual(database.Data, {})
            self.assertEqual(database.MetaData, {})
            self.assertEqual(len(caught), 1)
            backups = list(Path(directory).glob('analysis.json.corrupt-*'))
            self.assertEqual(len(backups), 1)
            self.assertFalse(database_path.exists())

    def test_update_atomically_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory, 'analysis.json')
            database = DB(database_path, 'default')
            database.setMetadata({'totalExperiments': 3})

            with database_path.open() as stream:
                contents = json.load(stream)
            self.assertEqual(contents['MetaData']['totalExperiments'], 3)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
