# -*- coding: utf-8 -*-
"""Example of sequence alignment using various methods.

Uses Needleman–Wunsch [#NW]_, Smith-Waterman [#SW]_, and Gotoh [#G]_
algorithms for sequence alignment.

.. [#NW] `A general method applicable to the search for similarities in the amino acid sequence of two proteins
          <http://dx.doi.org/10.1016%2F0022-2836%2870%2990057-4>`_.
.. [#SW] `Identification of common molecular subsequences
          <http://dx.doi.org/10.1016%2F0022-2836%2881%2990087-5>`_.
.. [#G] `An improved algorithm for matching biological sequences
         <http://dx.doi.org/10.1016/0022-2836(82)90398-9>`_.


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
                                   NW - Needleman-Wunsch (global),
                                   SW - Smith-Waterman (local),
                                   GG - Gotoh (global),
                                   GL - Gotoh (local)

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


def align(A, B, method=None, penalties=None):
    """Align sequence pair using one of the selected algorithms."""

    try:
        if not (isinstance(A, basestring) and isinstance(B, basestring)):
            raise ParameterError('Sequence provided is not a string.')
    except NameError:
        if not (isinstance(A, str) and isinstance(B, str)):
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

    if method not in ['NW', 'SW', 'GG', 'GL', None]:
        raise UnknownAlgorithmError('Unrecognized algorithm selection.')
    method = 'NW' if method is None else method

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

    if method == 'NW' or method == 'GG':
        # Initialize first row/column scores
        score[0, :, 0] = range(0, -len(col_idx), -1)
        score[0, :, 1] = 1
        score[:, 0, 0] = range(0, -len(row_idx), -1)
        score[:, 0, 3] = 1

    if method == 'GG' or method == 'GL':
        I = np.copy(score[:, :, 0])
        D = np.copy(score[:, :, 0])

    # Fill table with scores and arrows
    def fill_cell(r, c):
        match_score = penalties['match'] if col_idx[c] == row_idx[r] else penalties['mismatch']

        # Needleman–Wunsch
        if method == 'NW':
            diag = score[r - 1, c - 1, 0] + match_score # match/mismatch
            left = score[r, c - 1, 0] + penalties['indel'] # insertion
            top = score[r - 1, c, 0] + penalties['indel'] # deletion
            max_score = max([diag, left, top])

        # Smith-Waterman
        if method == 'SW':
            diag = score[r - 1, c - 1, 0] + match_score # match/mismatch
            left = score[r, c - 1, 0] + penalties['indel'] # insertion
            for l in range(2, c + 1):
                left = max(left, score[r, c - l, 0] + l * penalties['indel'])
            top = score[r - 1, c, 0] + penalties['indel'] # deletion
            for k in range(2, r + 1):
                top = max(top, score[r - k, c, 0] + k * penalties['indel'])
            max_score = max([diag, left, top, 0])

        # Gotoh
        if method == 'GG' or method == 'GL':
            if r == 1:
                I[r, c] = score[r, c - 1, 0] + penalties['gap_opening']
            else:
                I[r, c] = max(score[r, c - 1, 0] + penalties['gap_opening'], I[r, c - 1] + penalties['indel'])
            if c == 1:
                D[r, c] = score[r - 1, c, 0] + penalties['gap_opening']
            else:
                D[r, c] = max(score[r - 1, c, 0] + penalties['gap_opening'], D[r - 1, c] + penalties['indel'])
            diag = score[r - 1, c - 1, 0] + match_score # match/mismatch
            left = I[r, c] # insertion
            top = D[r, c] # deletion
        if method == 'GG':
            max_score = max([diag, left, top])
        if method == 'GL':
            max_score = max([diag, left, top, 0])

        score[r, c, 0] = max_score

        score[r, c, 1] = 1 if left == score[r, c, 0] else 0
        score[r, c, 2] = 1 if diag == score[r, c, 0] else 0
        score[r, c, 3] = 1 if top == score[r, c, 0] else 0

    for row in range(1, len(row_idx)):
        for col in range(1, len(col_idx)):
            fill_cell(row, col)

    if method == 'NW' or method == 'GG':
        row = len(row_idx) - 1
        col = len(col_idx) - 1
    if method == 'SW' or method == 'GL':
        row, col = np.unravel_index(score[:, :, 0].argmax(), np.shape(score[:, :, 0]))
    final_score = score[row, col, 0]

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

    while True:
        row, col = trace_cell(row, col)
        if method == 'NW' or method == 'GG':
            if row == 0 and col == 0:
                break
        if method == 'SW' or method == 'GL':
            if score[row, col, 0] == 0:
                break

    return final_score, ''.join(result_A), ''.join(result_B)


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
        'indel': int(arguments['--indel']),
    }

    if arguments['--opening'] is not None:
        penalties['gap_opening'] = int(arguments['--opening'])

    score, result_A, result_B = align(A, B, method, penalties)

    if arguments['--output'] is not None:
        try:
            with open(arguments['--output'], 'w') as output_file:
                output_file.write('score;first-sequence;second-sequence\n%d;%s;%s' % (score, result_A, result_B))
        except:
            raise ParameterError('Failed saving results to file.')
    else:
        print('Score:\n%d\n' % score)
        print('Aligned sequences:')
        print(result_A)
        print(result_B)