# -*- coding: utf-8 -*-
"""Test suite for sequence alignment implementations."""

import unittest
import numpy as np
import time
import subprocess
import os
import string
import shlex
import sys
import inspect

from src import aligner


class TestAligner(unittest.TestCase):
    """Various tests for correctness of the underlying implementation."""

    def setUp(self):
        np.random.seed(20150516)

    def _generate_random_sequence(self, length, alphabet):
        return list((np.random.choice(alphabet) for _ in range(length)))

    def test_aligner_should_provide_Smith_Waterman_algorithm(self):
        score, A, B = aligner.align('ACACACTA', 'AGCACACA', 'SW', {'match': 2, 'mismatch': -1, 'indel': -1})
        self.assertEqual(score, 12)
        self.assertIn((A, B), [('A-CACACTA', 'AGCACAC-A')])

        score, A, B = aligner.align('CGTGAATTCAT', 'GACTTAC', 'SW', {'match': 5, 'mismatch': -3, 'indel': -4})
        self.assertEqual(score, 18)
        self.assertIn((A, B), [('GAATTCA', 'GACTT-A'), ('GAATT-C', 'GACTTAC')])

    def test_aligner_should_provide_Gotoh_algorithm(self):
        score, A, B = aligner.align('ACACACTA', 'AGCACACA', 'GL', {'match': 2, 'mismatch': -1, 'indel': -1})
        self.assertEqual(score, 12)
        self.assertIn((A, B), [('A-CACACTA', 'AGCACAC-A')])

        score, A, B = aligner.align('CGTGAATTCAT', 'GACTTAC', 'GL', {'match': 5, 'mismatch': -3, 'indel': -4})
        self.assertEqual(score, 18)
        self.assertIn((A, B), [('GAATTCA', 'GACTT-A'), ('GAATT-C', 'GACTTAC')])

        score, A, B = aligner.align('CGGTCATAC', 'CGGAT', 'GG', {'match': 1, 'mismatch': -1, 'indel': -1, 'gap_opening': -5})
        self.assertEqual(score, -5)
        self.assertIn((A, B), [('CGGTCATAC', 'CGG----AT')])

    def test_aligner_should_raise_UnknownAlgorithmError_on_call_with_unknown_method(self):
        with self.assertRaises(aligner.UnknownAlgorithmError):
            aligner.align('GCATGCU', 'GATTACA', method='XY')

    def test_aligner_should_accept_custom_penalties(self):
        score, A, B = aligner.align('GCATGCU', 'GCATGCU', None, {'match': 10, 'mismatch': -1, 'indel': -1}) # 7 matches
        self.assertEqual(score, 70)

        score, A, B = aligner.align('GCATGCU', 'GTATGAG', None, {'match': 0, 'mismatch': -3, 'indel': -10}) # 3 mutations
        self.assertEqual(score, -9)

        score, A, B = aligner.align('GCTGCU', 'GCATGC', None, {'match': 0, 'mismatch': -10, 'indel': -4}) # 2 deletions
        self.assertEqual(score, -8)

    def test_aligner_should_work_on_Wikipedia_examples(self):
        score, A, B = aligner.align('GCATGCU', 'GATTACA', None)
        self.assertEqual(score, 0)
        self.assertIn((A, B), [('GCATG-CU', 'G-ATTACA'), ('GCA-TGCU', 'G-ATTACA'), ('GCAT-GCU', 'G-ATTACA')]) # Accept one of possible variants

    def test_aligner_should_detect_mutations(self, sequence_length=100):
        alphabet = ['A', 'C', 'G', 'T']

        A = self._generate_random_sequence(sequence_length, alphabet)
        B = list(A)

        mutation_count = 0
        for i in range(len(A)):
            if np.random.random() < 0.2:
                mutation = alphabet[np.random.randint(len(alphabet))]
                if B[i] != mutation:
                    mutation_count = mutation_count + 1
                    B[i] = mutation

        score, _, _ = aligner.align(''.join(A), ''.join(A), None)
        self.assertEqual(score, sequence_length)

        score, _, _ = aligner.align(''.join(A), ''.join(B), None, {'match': 1, 'mismatch': -1, 'indel': -10})
        self.assertEqual(score, sequence_length + mutation_count * -2)

    def test_aligner_should_detect_deletions(self, sequence_length=100):
        alphabet = ['A', 'C', 'G', 'T']

        A = self._generate_random_sequence(sequence_length, alphabet)
        B = []

        deletion_count = 0
        for i in range(len(A)):
            if np.random.random() < 0.2:
                deletion_count = deletion_count + 1
            else:
                B.append(A[i])

        score, _, _ = aligner.align(''.join(A), ''.join(A), None)
        self.assertEqual(score, sequence_length)

        score, _, _ = aligner.align(''.join(A), ''.join(B), None, {'match': 1, 'mismatch': -100, 'indel': -1})
        self.assertEqual(score, sequence_length + deletion_count * -2)

    def test_aligner_should_detect_insertions(self, sequence_length=100):
        alphabet = ['A', 'C', 'G', 'T']

        A = self._generate_random_sequence(sequence_length, alphabet)
        B = []

        insertion_count = 0
        for i in range(len(A)):
            B.append(A[i])
            if np.random.random() < 0.2:
                insertion_count = insertion_count + 1
                B.append(alphabet[np.random.randint(len(alphabet))])

        score, _, _ = aligner.align(''.join(A), ''.join(A), None)
        self.assertEqual(score, sequence_length)

        score, _, _ = aligner.align(''.join(A), ''.join(B), None, {'match': 1, 'mismatch': -100, 'indel': -1})
        self.assertEqual(score, sequence_length + insertion_count + insertion_count * -2) # Insertions also increase total sequence length

    #@unittest.skip('Skipping long tests.')
    def test_aligner_should_work_on_long_sequences(self):
        start = time.time()
        self.test_aligner_should_detect_mutations(sequence_length=1000)
        total = time.time() - start
        self.assertLess(total, 30)

        start = time.time()
        self.test_aligner_should_detect_insertions(sequence_length=1000)
        total = time.time() - start
        self.assertLess(total, 30)

        start = time.time()
        self.test_aligner_should_detect_deletions(sequence_length=1000)
        total = time.time() - start
        self.assertLess(total, 30)

    @unittest.skip('Not implemented at this moment. Requires RPy bindings.')
    def test_aligner_should_give_same_results_as_biostrings(self):
        raise NotImplementedError()

    @unittest.skip('Not implemented at this moment. Requires Biopython.')
    def test_aligner_should_give_same_results_as_biopython(self):
        raise NotImplementedError()

    def _run_CLI(self, args):
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../src/aligner.py'
        try:
            return subprocess.check_output([sys.executable] + [path] + shlex.split(args), stderr=subprocess.STDOUT, env=os.environ.copy())
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise e

    def _parse_CLI_output(self, out):
        out = out.split('\n')
        score = int(out[1])
        A = out[4].strip()
        B = out[5].strip()
        return score, A, B

    def test_CLI_should_accept_sequences_as_arguments(self):
        score, A, B = self._parse_CLI_output(self._run_CLI('GCATGCU GATTACA'))
        self.assertEqual(score, 0)
        self.assertIn((A, B), [('GCATG-CU', 'G-ATTACA'), ('GCA-TGCU', 'G-ATTACA'), ('GCAT-GCU', 'G-ATTACA')]) # Accept one of possible variants

    def test_CLI_should_accept_custom_penalties(self):
        score, A, B = self._parse_CLI_output(self._run_CLI('--match=10 --mismatch=-1 --indel=-1 GCATGCU GCATGCU')) # 7 matches
        self.assertEqual(score, 70)

        score, A, B = self._parse_CLI_output(self._run_CLI('--match=0 --mismatch=-3 --indel=-10 GCATGCU GTATGAG')) # 3 mutations
        self.assertEqual(score, -9)

        score, A, B = self._parse_CLI_output(self._run_CLI('--match=0 --mismatch=-10 --indel=-4 GCTGCU GCATGC')) # 2 deletions
        self.assertEqual(score, -8)

    def test_CLI_should_accept_sequences_from_files(self):
        tmp_dir = '_tmp_' + ''.join(np.random.choice(list(string.ascii_uppercase + string.digits)) for _ in range(8))
        try:
            os.mkdir(tmp_dir)
            with open('{}/A.txt'.format(tmp_dir), 'w') as sequence_file:
                sequence_file.write('GCATGCU')
            with open('{}/B.txt'.format(tmp_dir), 'w') as sequence_file:
                sequence_file.write('GATTACA')

            score, A, B = self._parse_CLI_output(self._run_CLI('-i {0}/A.txt {0}/B.txt'.format(tmp_dir)))
            self.assertEqual(score, 0)
            self.assertIn((A, B), [('GCATG-CU', 'G-ATTACA'), ('GCA-TGCU', 'G-ATTACA'), ('GCAT-GCU', 'G-ATTACA')]) # Accept one of possible variants
        finally:
            try:
                os.unlink('{}/A.txt'.format(tmp_dir))
                os.unlink('{}/B.txt'.format(tmp_dir))
            except OSError:
                pass
            os.rmdir(tmp_dir)

    def test_CLI_should_save_result_to_file(self):
        tmp_dir = '_tmp_' + ''.join(np.random.choice(list(string.ascii_uppercase + string.digits)) for _ in range(8))
        try:
            os.mkdir(tmp_dir)
            self._run_CLI('-o {}/results.csv GCATGCU GATTACA'.format(tmp_dir))
            with open('{}/results.csv'.format(tmp_dir), 'r') as output_file:
                score, A, B = output_file.read().split('\n')[1].split(';')

            self.assertEqual(int(score), 0)
            self.assertIn((A, B), [('GCATG-CU', 'G-ATTACA'), ('GCA-TGCU', 'G-ATTACA'), ('GCAT-GCU', 'G-ATTACA')]) # Accept one of possible variants
        finally:
            try:
                os.unlink('{}/results.csv'.format(tmp_dir))
            except OSError:
                pass
            os.rmdir(tmp_dir)

    def test_CLI_should_provide_different_algorithms(self):
        score, A, B = self._parse_CLI_output(self._run_CLI('--match=2 --mismatch=-1 --indel=-1 --method=SW ACACACTA AGCACACA'))
        self.assertEqual(score, 12)
        self.assertIn((A, B), [('A-CACACTA', 'AGCACAC-A')])

        score, A, B = self._parse_CLI_output(self._run_CLI('--match=5 --mismatch=-3 --indel=-4 --method=SW CGTGAATTCAT GACTTAC'))
        self.assertEqual(score, 18)
        self.assertIn((A, B), [('GAATTCA', 'GACTT-A'), ('GAATT-C', 'GACTTAC')])

        score, A, B = self._parse_CLI_output(self._run_CLI('--match=2 --mismatch=-1 --indel=-1 --method=GL ACACACTA AGCACACA'))
        self.assertEqual(score, 12)
        self.assertIn((A, B), [('A-CACACTA', 'AGCACAC-A')])

        score, A, B = self._parse_CLI_output(self._run_CLI('--match=5 --mismatch=-3 --indel=-4 --method=GL CGTGAATTCAT GACTTAC'))
        self.assertEqual(score, 18)
        self.assertIn((A, B), [('GAATTCA', 'GACTT-A'), ('GAATT-C', 'GACTTAC')])

        score, A, B = self._parse_CLI_output(self._run_CLI('--match=1 --mismatch=-1 --indel=-1 --opening=-5 --method=GG CGGTCATAC CGGAT'))
        self.assertEqual(score, -5)
        self.assertIn((A, B), [('CGGTCATAC', 'CGG----AT')])


if __name__ == '__main__':
    unittest.main()