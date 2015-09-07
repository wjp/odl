# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
Utilities for use inside the ODL project, not for external use.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str

# External module imports
from textwrap import dedent, fill
import numpy as np


def errfmt(errstr):
    """TODO: remove"""
    return fill(dedent(errstr)).lstrip()


def array1d_repr(array):
    """Stringification of a 1D array, keeping byte / unicode."""
    if len(array) < 7:
        return repr(list(array[:]))
    else:
        return (repr(list(array[:3])).rstrip(']') + ', ..., ' +
                repr(list(array[-3:])).lstrip('['))


def array1d_str(array):
    """Stringification of a 1D array, regardless of byte or unicode."""
    if len(array) < 7:
        inner_str = ', '.join(str(a) for a in array)
        return '[{}]'.format(inner_str)
    else:
        left_str = ', '.join(str(a) for a in array[:3])
        right_str = ', '.join(str(a) for a in array[-3:])
        return '[{}, ..., {}]'.format(left_str, right_str)


def dtype_repr(dtype):
    """Stringification of data type with default for `int` and `float`."""
    if dtype == np.dtype(int):
        return "'int'"
    elif dtype == np.dtype(float):
        return "'float'"
    else:
        return "'{}'".format(dtype)

if __name__ == '__main__':
    import doctest
    doctest.testmod()