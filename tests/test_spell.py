import unittest
import re
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
from spell.spell import LogParser, LCSObject, Node

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

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
DF_MOCK = pd.DataFrame(mock)

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
        df_log = self.parser.log_to_dataframe(
            test_data_path, regex, header, LOG_FORMAT
        )
        assert_frame_equal(df_log, DF_MOCK)

    def test_load_data(self):
        self.parser.logformat = LOG_FORMAT
        self.parser.path = THIS_DIR
        self.parser.logname = 'test_data.log'
        self.parser.load_data()
        assert_frame_equal(self.parser.df_log, DF_MOCK)

    def test_addSeqToPrefixTree(self):
        logmessageL = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.19.102', '54106', 'dest', '/10.250.19.102', '50010']
        logID = 0

        rootNode = Node()
        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])

        self.parser.addSeqToPrefixTree(rootNode, newCluster)
        res = helper(rootNode)
        self.assertEqual(res, logmessageL)

    def test_LCS(self):
        seq1 = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.10.6', '40524', 'dest', '/10.250.10.6', '50010']
        seq2 = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.19.102', '54106', 'dest', '/10.250.19.102', '50010']
        expected_lcs = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', 'dest', '50010']

        lcs = self.parser.LCS(seq1, seq2)
        self.assertListEqual(lcs, expected_lcs)


def helper(rootNode):
    if rootNode.childD == dict():
        return []

    res = []
    for k in rootNode.childD.keys():
        res.append(k)
        res += helper(rootNode.childD[k])
    return res


if __name__ == '__main__':
    unittest.main()
