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
    aligner.py [options] -i <first-sequence-file> <second-sequence-file>

Arguments:
    first-sequence   Text sequence used for pairwise comparison.
    second-sequence  Text sequence used for pairwise comparison.

Options:
    -h --help
    --version

    -i                         Loads sequences from files.
    -o FILE --output=FILE      Saves result to FILE.
    -m METHOD --method=METHOD  Selects one of the implemented alignment algorithms [default: NW]:
                                   NW - Needleman-Wunsch,
                                   SW - Smith-Waterman,
                                   GL - Gotoh (local),
                                   GG - Gotoh (global),
                                   AE - Altschul-Erickson.

    --match=PENALTY     Score for matching letters [default: 1].
    --mismatch=PENALTY  Score for different letters [default: -1].
    --indel=PENALTY     Score for INsertion/DELetion (gap) [default: -1].
    --opening=PENALTY   Score for new gap opening (default: same as INDEL). Works with Gotoh method.

"""

from docopt import docopt
import numpy as np


class UnknownAlgorithmError(Exception):
    pass


class ParameterError(Exception):
    pass


def _trace_cell(r, c, score, col_idx, row_idx, result_A, result_B):
    """Trace arrows back to origin cell."""

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


def _init_align(A, B):
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

    return result_A, result_B, col_idx, row_idx, score


def _align(A, B, penalties, mode='global'):
    result_A, result_B, col_idx, row_idx, score = _init_align(A, B)

    if mode == 'global':
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

        if mode == 'global':
            max_score = max([diag, left, top])
        else:
            max_score = max([diag, left, top, 0])

        score[r, c, 0] = max_score

        score[r, c, 1] = 1 if left == score[r, c, 0] else 0
        score[r, c, 2] = 1 if diag == score[r, c, 0] else 0
        score[r, c, 3] = 1 if top == score[r, c, 0] else 0

    for row in range(1, len(row_idx)):
        for col in range(1, len(col_idx)):
            fill_cell(row, col)

    if mode == 'global':
        row = len(row_idx) - 1
        col = len(col_idx) - 1
    else:
        row, col = np.unravel_index(score[:, :, 0].argmax(), np.shape(score[:, :, 0]))
    final_score = score[row, col, 0]

    while True:
        row, col = _trace_cell(row, col, score, col_idx, row_idx, result_A, result_B)
        if mode == 'global':
            if row == 0 and col == 0:
                break
        else:
            if score[row, col, 0] == 0:
                break

    return final_score, ''.join(result_A), ''.join(result_B)


def align_nw(A, B, penalties):
    """Global align sequence pair using Needleman-Wunsch algorithm."""

    return _align(A, B, penalties, 'global')


def align_sw(A, B, penalties):
    """Local align sequence pair using Smith-Waterman algorithm."""

    return _align(A, B, penalties, 'local')


def align_g(A, B, penalties, mode='global'):
    """Align sequence pair using Gotoh algorithm with support for exlicit gap opening penalty."""

    result_A, result_B, col_idx, row_idx, score = _init_align(A, B)

    if mode == 'global':
        # Initialize first row/column scores
        score[0, :, 0] = range(0, -len(col_idx), -1)
        score[0, :, 1] = 1
        score[:, 0, 0] = range(0, -len(row_idx), -1)
        score[:, 0, 3] = 1

    D = np.copy(score[:, :, 0])
    I = np.copy(score[:, :, 0])

    # Fill table with scores and arrows
    def fill_cell(r, c):
        if r == 1:
            I[r, c] = score[r, c - 1, 0] + penalties['gap_opening']
        else:
            I[r, c] = max(score[r, c - 1, 0] + penalties['gap_opening'], I[r, c - 1] + penalties['indel'])

        if c == 1:
            D[r, c] = score[r - 1, c, 0] + penalties['gap_opening']
        else:
            D[r, c] = max(score[r - 1, c, 0] + penalties['gap_opening'], D[r - 1, c] + penalties['indel'])

        alignment_penalty = penalties['match'] if col_idx[c] == row_idx[r] else penalties['mismatch']
        diag = score[r - 1, c - 1, 0] + alignment_penalty
        left = I[r, c]
        top = D[r, c]

        if mode == 'global':
            max_score = max([diag, left, top])
        else:
            max_score = max([diag, left, top, 0])
        score[r, c, 0] = max_score

        score[r, c, 1] = 1 if left == score[r, c, 0] else 0
        score[r, c, 2] = 1 if diag == score[r, c, 0] else 0
        score[r, c, 3] = 1 if top == score[r, c, 0] else 0

    for row in range(1, len(row_idx)):
        for col in range(1, len(col_idx)):
            fill_cell(row, col)

    if mode == 'global':
        row = len(row_idx) - 1
        col = len(col_idx) - 1
    else:
        row, col = np.unravel_index(score[:, :, 0].argmax(), np.shape(score[:, :, 0]))
    final_score = score[row, col, 0]

    while True:
        row, col = _trace_cell(row, col, score, col_idx, row_idx, result_A, result_B)
        if mode == 'global':
            if row == 0 and col == 0:
                break
        else:
            if score[row, col, 0] == 0:
                break

    return final_score, ''.join(result_A), ''.join(result_B)


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

    try:
        penalties['gap_opening']
    except KeyError:
        penalties['gap_opening'] = penalties['indel']

    if method not in ['NW', 'SW', 'GG', 'GL', 'AE', None]:
        raise UnknownAlgorithmError('Unrecognized algorithm selection.')
    method = 'NW' if method is None else method

    if method == 'NW':
        return align_nw(A, B, penalties)

    if method == 'SW':
        return align_sw(A, B, penalties)

    if method == 'GG':
        return align_g(A, B, penalties, 'global')

    if method == 'GL':
        return align_g(A, B, penalties, 'local')

    if method == 'AE':
        return align_ae(A, B, penalties)


if __name__ == '__main__':
    arguments = docopt(__doc__, help=True, version='Projekt / Metody Bioinformatyki - 2015-05-16', options_first=True)

    if arguments['-i']:
        A_file, B_file = (arguments['<first-sequence-file>'], arguments['<second-sequence-file>'])
        try:
            with open(A_file, 'r') as sequence_file:
                A = sequence_file.read()
            with open(B_file, 'r') as sequence_file:
                B = sequence_file.read()
            if not len(A) or not len(B):
                raise ParameterError('Failed loading sequence files.')
        except:
            raise ParameterError('Failed loading sequence files.')
    else:
        A, B = (arguments['<first-sequence>'], arguments['<second-sequence>'])

    method = arguments['--method']

    penalties = {
        'match': int(arguments['--match']),
        'mismatch': int(arguments['--mismatch']),
        'indel': int(arguments['--indel'])
    }

    score, result_A, result_B = align(A, B, method, penalties)

    if arguments['--output'] is not None:
        try:
            with open(arguments['--output'], 'w') as output_file:
                output_file.write('score;first-sequence;second-sequence\n%d;%s;%s' % (score, result_A, result_B))
        except:
            raise ParameterError('Failed saving results to file.')
    else:
        print 'Score:\n%d' % score
        print '\nAligned sequences:\n', result_A, '\n', result_B