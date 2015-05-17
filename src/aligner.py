# -*- coding: utf-8 -*-
"""Example of sequence alignment using Gotoh and Altschul-Erickson methods.

Uses Needleman–Wunsch [#NW]_, Smith-Waterman [#SW]_, Gotoh [#G]_, and Altschul-Erickson [#AE]_
algorithms for sequence alignment.

.. [#NW] `A general method applicable to the search for similarities in the amino acid sequence of two proteins
          <http://dx.doi.org/10.1016%2F0022-2836%2870%2990057-4>`_.
.. [#SW] http://dx.doi.org/10.1016%2F0022-2836%2881%2990087-5
.. [#G] http://dx.doi.org/10.1016/0022-2836(82)90398-9
.. [#AE] http://dx.doi.org/10.1007%2FBF02462326


Usage:
    aligner.py [options] <first-sequence> <second-sequence>

Arguments:
    first-sequence   Text sequence used for pairwise comparison.
    second-sequence  Text sequence used for pairwise comparison.

Options:
    -h --help
    --version

    -m METHOD --method=METHOD  Selects one of the implemented alignment algorithms [default: NW]:
                                   NW - Needleman-Wunsch,
                                   SW - Smith-Waterman,
                                   G  - Gotoh,
                                   AE - Altschul-Erickson.

    --match=PENALTY     Score for matching letters [default: 1].
    --mismatch=PENALTY  Score for different letters [default: -1].
    --indel=PENALTY     Score for INsertion/DELetion (gap) [default: -1].

"""

from docopt import docopt
import numpy as np

class UnknownAlgorithmError(Exception):
    pass


class ParameterError(Exception):
    pass


def align_nw(A, B, penalties):
    """Align sequence pair using Needleman-Wunsch algorithm."""

    result_A = []
    result_B = []

    # Top row/left column (headers)
    col_idx = ' ' + A
    row_idx = ' ' + B

    # Create a 3D score matrix with following axes:
    #   - rows
    #   - columns
    #   - cell values holder: [cell score, left arrow, diagonal arrow, top arrow]
    score = np.zeros((len(row_idx), len(col_idx), 4))

    # Initialize first row/column scores
    score[0, :, 0] = range(0, -len(col_idx), -1)
    score[0, :, 1] = 1
    score[:, 0, 0] = range(0, -len(row_idx), -1)
    score[:, 0, 3] = 1

    # Fill table with scores and arrows
    def fill_cell(r, c):
        alignment_penalty = penalties['match'] if col_idx[c] == row_idx[r] else penalties['mismatch']
        diag = score[r - 1, c - 1, 0] + alignment_penalty
        left = score[r, c - 1, 0] + penalties['indel']
        top = score[r - 1, c, 0] + penalties['indel']

        max_score = max([diag, left, top])
        score[r, c, 0] = max_score

        score[r, c, 1] = 1 if left == score[r, c, 0] else 0
        score[r, c, 2] = 1 if diag == score[r, c, 0] else 0
        score[r, c, 3] = 1 if top == score[r, c, 0] else 0

    for row in range(1, len(row_idx)):
        for col in range(1, len(col_idx)):
            fill_cell(row, col)

    # Trace arrows back to origin cell
    def trace_cell(r, c):
        A_letter = col_idx[c]
        B_letter = row_idx[r]

        if score[r, c, 1]: # horizontal arrow
            result_A.insert(0, A_letter)
            result_B.insert(0, '-')
            return r, c -1

        if score[r, c, 2]: # diagonal arrow
            result_A.insert(0, A_letter)
            result_B.insert(0, B_letter)
            return r - 1, c - 1

        if score[r, c, 3]: # vertical arrow
            result_A.insert(0, '-')
            result_B.insert(0, B_letter)
            return r - 1, c

    row = len(row_idx) - 1
    col = len(col_idx) - 1
    final_score = score[row, col, 0]

    while True:
        row, col = trace_cell(row, col)
        if row == 0 and col == 0:
            break

    return final_score, ''.join(result_A), ''.join(result_B)


def align_sw(A, B, penalties):
    """Align sequence pair using Smith-Waterman algorithm."""
    raise NotImplementedError()


def align_g(A, B, penalties):
    """Align sequence pair using Gotoh algorithm."""
    raise NotImplementedError()


def align_ae(A, B, penalties):
    """Align sequence pair using Altschul-Erickson algorithm."""
    raise NotImplementedError()


def align(A, B, method=None, penalties=None):
    """Align sequence pair using one of the selected algorithms."""

    if not (isinstance(A, basestring) and isinstance(B, basestring)):
        raise ParameterError('Sequence provided is not a string.')

    if penalties is None:
        penalties = {'match': 1, 'mismatch': -1, 'indel': -1} # default N-W scoring
    try:
        penalties['match']
        penalties['mismatch']
        penalties['indel']
    except KeyError:
        raise ParameterError('Malformatted penalty dictionary.')

    if method not in ['NW', 'SW', 'G', 'AE', None]:
        raise UnknownAlgorithmError('Unrecognized algorithm selection.')
    method = 'NW' if method is None else method

    if method == 'NW':
        return align_nw(A, B, penalties)

    if method == 'SW':
        return align_sw(A, B, penalties)

    if method == 'G':
        return align_g(A, B, penalties)

    if method == 'AE':
        return align_ae(A, B, penalties)


if __name__ == '__main__':
    arguments = docopt(__doc__, help=True, version='Projekt / Metody Bioinformatyki - 2015-05-16', options_first=True)

    A, B = (arguments['<first-sequence>'], arguments['<second-sequence>'])
    if not (isinstance(A, basestring) and isinstance(B, basestring)):
        raise ParameterError('Sequence provided is not a string.')

    method = arguments['--method']
    if method not in ['NW', 'SW', 'G', 'AE', None]:
        raise UnknownAlgorithmError('Unrecognized algorithm selection.')

    penalties = {
        'match': int(arguments['--match']),
        'mismatch': int(arguments['--mismatch']),
        'indel': int(arguments['--indel'])
    }

    score, result_A, result_B = align(A, B, method, penalties)

    print 'Score:\n%d' % score
    print '\nAligned sequences:\n', result_A, '\n', result_B