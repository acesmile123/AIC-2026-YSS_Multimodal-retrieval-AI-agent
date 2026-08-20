import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.qa_cache import LRUCache, answer_cache_key
from qa.qa_answer_normalizer import parse_vlm_output
from qa.qa_candidate_fusion import fuse_candidate_lists
from qa.qa_types import QACandidate


class CoreTests(unittest.TestCase):
    def test_cache_eviction(self):
        cache = LRUCache(max_size=2)
        cache.set('a', 1); cache.set('b', 2); cache.get('a'); cache.set('c', 3)
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('a'), 1)
        self.assertEqual(cache.get('c'), 3)

    def test_answer_cache_key_is_stable(self):
        self.assertEqual(answer_cache_key('v', 1, ' q '), answer_cache_key('v', 1, 'q'))

    def test_vlm_parser(self):
        answer, conf = parse_vlm_output('ANSWER: màu xanh\nCONFIDENCE: HIGH', max_words=4)
        self.assertEqual(answer, 'màu xanh')
        self.assertEqual(conf, 1.0)

    def test_vlm_parser_fallback(self):
        answer, conf = parse_vlm_output('màu đỏ', max_words=4)
        self.assertEqual(answer, 'màu đỏ')
        self.assertGreater(conf, 0.0)

    def test_rrf_fusion(self):
        a = [QACandidate('v1', 10, 1.0), QACandidate('v2', 20, 0.5)]
        b = [QACandidate('v1', 10, 0.8)]
        result = fuse_candidate_lists([a, b], weights=[1.0, 1.0])
        self.assertEqual((result[0].video_id, result[0].frame_id), ('v1', 10))


if __name__ == '__main__':
    unittest.main()
