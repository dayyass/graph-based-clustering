import unittest

from graph_clustering.check import check_adjacency_matrix, check_symmetric

from .test_utils import adjacency_matrix, distances


class TestCheck(unittest.TestCase):
    def test_check_symmetric(self):
        """Test check_symmetric"""

        self.assertTrue(check_symmetric(distances))

        self.assertTrue(check_symmetric(adjacency_matrix))

    def test_check_adjacency_matrix(self):
        """Test check_adjacency_matrix"""

        self.assertFalse(check_adjacency_matrix(distances))

        self.assertTrue(check_adjacency_matrix(adjacency_matrix))
