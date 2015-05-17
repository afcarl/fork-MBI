# -*- coding: utf-8 -*-
"""Test suite for sequence alignment implementations."""

import unittest
import numpy as np
import time
from src import aligner

class TestAligner(unittest.TestCase):
    """Various tests for correctness of the underlying implementation."""

    def setUp(self):
        np.random.seed(20150516)

    def _generate_random_sequence(self, length, alphabet):
        return list((np.random.choice(alphabet) for _ in range(length)))

    def test_aligner_should_provide_Needleman_Wunsch_algorithm(self):
        self.test_aligner_should_work_on_Wikipedia_examples(method='NW')

    def test_aligner_should_provide_Smith_Waterman_algorithm(self):
        self.test_aligner_should_work_on_Wikipedia_examples(method='SW')

    def test_aligner_should_provide_Altschul_Erickson_algorithm(self):
        self.test_aligner_should_work_on_Wikipedia_examples(method='AE')

    def test_aligner_should_provide_Gotoh_algorithm(self):
        self.test_aligner_should_work_on_Wikipedia_examples(method='G')

    def test_aligner_should_raise_UnknownAlgorithmError_on_call_with_unknown_method(self):
        with self.assertRaises(aligner.UnknownAlgorithmError):
            aligner.align('GCATGCU', 'GATTACA', method='XY')

    def test_aligner_should_accept_custom_penalties(self):
        raise NotImplementedError()

    def test_aligner_should_work_on_Wikipedia_examples(self, method=None):
        score, A, B = aligner.align('GCATGCU', 'GATTACA', method)
        self.assertEqual(score, 0)
        self.assertIn((A, B), [('GCATG-CU', 'G-ATTACA'), ('GCA-TGCU', 'G-ATTACA'), ('GCAT-GCU', 'G-ATTACA')]) # Accept one of possible variants

    def test_aligner_should_detect_mutations(self, sequence_length=100):
        alphabet = ['A', 'C', 'G', 'T']

        A = self._generate_random_sequence(sequence_length, alphabet)
        B = list(A)

        mutation_count = 0
        for i in range(len(B)):
            if np.random.random() < 0.2:
                mutation = alphabet[np.random.randint(len(alphabet))]
                if B[i] != mutation:
                    mutation_count = mutation_count + 1
                    B[i] = mutation

        score, _, _ = aligner.align(''.join(A), ''.join(A))
        self.assertEqual(score, sequence_length)

        score, _, _ = aligner.align(''.join(A), ''.join(B), None, {'match': 1, 'mismatch': -1, 'indel': -10})
        self.assertEqual(score, sequence_length + mutation_count * -2)

    def test_aligner_should_detect_deletions(self):
        raise NotImplementedError()

    def test_aligner_should_detect_insertions(self):
        raise NotImplementedError()

    def test_aligner_should_work_on_long_sequences(self):
        start = time.time()
        self.test_aligner_should_detect_mutations(sequence_length=1000)
        total = time.time() - start
        self.assertLess(total, 30)

    def test_aligner_should_give_same_results_as_biostrings(self):
        raise NotImplementedError()

    def test_aligner_should_give_same_results_as_biopython(self):
        raise NotImplementedError()

    def test_CLI_should_accept_sequences_as_arguments(self):
        raise NotImplementedError()

    def test_CLI_should_accept_sequences_from_files(self):
        raise NotImplementedError()

    def test_CLI_should_accept_custom_penalty_matrix(self):
        raise NotImplementedError()

    def test_CLI_should_save_result_to_file(self):
        raise NotImplementedError()

    def test_CLI_should_provide_different_algorithms(self):
        raise NotImplementedError()


if __name__ == '__main__':
    unittest.main()