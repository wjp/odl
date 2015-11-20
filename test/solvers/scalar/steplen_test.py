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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str

# External module imports
import numpy as np
import pytest

# Internal module imports
import odl
from odl.util.testutils import almost_equal


def test_barzilai_borwein_step():
    n = 3

    # Np as validation
    A = np.random.rand(n, n) + np.eye(n) * n
    x = np.random.rand(n)
    b = np.dot(A, x)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.element(x)
    rhs = rn.element(b)

    # Make operator
    Aop = odl.MatVecOperator(rn, rn, A)

    # Operator for Barzilai-Borwein step
    step_param = odl.solvers.BarzilaiBorweinStep(
        lambda x: Aop.adjoint(Aop(x) - rhs))
    # Should return the default value if update is 0
    # Compute a step
    step = step_param(xvec, xvec)
    assert almost_equal(step, 0.0005)

    # Test #2
    Aop = odl.IdentityOperator(rn)
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    b = np.ones(n)

    x = rn.element(x)
    y = rn.element(y)
    rhs = rn.element(b)
    step_param = odl.solvers.BarzilaiBorweinStep(
        lambda x: Aop.adjoint(Aop(x) - rhs))
    step = step_param(y, x)
    # With such choices of x,y, and Aop step == 1.0
    assert almost_equal(step, 1.0)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
