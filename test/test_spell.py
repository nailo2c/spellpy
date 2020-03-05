import unittest
import re
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
from spell.spell import LogParser

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

class TestLogParser(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_generate_logformat_regex(self):
        expected_header = ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content']
        expected_regex = re.compile(
            '^(?P<Date>.*?) (?P<Time>.*?) (?P<Pid>.*?) (?P<Level>.*?) (?P<Component>.*?): (?P<Content>.*?)$'
        )

        header, regex = self.parser.generate_logformat_regex(LOG_FORMAT)
        self.assertListEqual(header, expected_header)
        self.assertCountEqual(header, expected_header)
        self.assertEqual(regex, expected_regex)

    def test_log_to_dataframe(self):
        test_data_path = os.path.join(THIS_DIR, 'test_data.log')
        header, regex = self.parser.generate_logformat_regex(LOG_FORMAT)
        log_df = self.parser.log_to_dataframe(
            test_data_path, regex, header, LOG_FORMAT
        )

        mock = {
            'LineId': [1, 2, 3],
            'Date': ['081109', '081109', '081109'],
            'Time': ['203518', '203518', '203519'],
            'Pid': ['143', '35', '143'],
            'Level': ['INFO', 'INFO', 'INFO'],
            'Component': ['dfs.DataNode$DataXceiver', 'dfs.FSNamesystem', 'dfs.DataNode$DataXceiver'],
            'Content': [
                'Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010',
                'BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906',
                'Receiving block blk_-1608999687919862906 src: /10.250.10.6:40524 dest: /10.250.10.6:50010'
            ],
        }
        expected_df = pd.DataFrame(mock)
        assert_frame_equal(log_df, expected_df)


if __name__ == '__main__':
    unittest.main()
