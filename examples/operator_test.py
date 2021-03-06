﻿# Copyright 2014, 2015 The ODL development group
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

"""Run the standardized test suite on a convolution operator."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np
import scipy
import scipy.ndimage

# Internal
import odl


class Convolution(odl.Operator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel

        # Scaling factor needed to account for non-unity cell volume
        self.scale = disc_kernel.space.cell_volume

        super().__init__(space, space, linear=True)

    def _apply(self, rhs, out):
        scipy.ndimage.convolve(rhs,
                               self.kernel,
                               output=out.asarray(),
                               mode='wrap')

        out *= self.scale

    @property
    def adjoint(self):
        return Convolution(self.domain, self.adjkernel, self.kernel)


def kernel(x, y):
    mean = [0.0, 0.25]
    std = [0.05, 0.05]
    return np.exp(-(((x-mean[0])/std[0])**2 + ((y-mean[1])/std[1])**2))


def adjkernel(x, y):
    return kernel(-x, -y)


# Continuous definition of problem
cont_space = odl.FunctionSpace(odl.Rectangle([-1, -1], [1, 1]))
kernel_space = odl.FunctionSpace(cont_space.domain - cont_space.domain)

# Complicated functions to check performance
cont_kernel = kernel_space.element(kernel)
cont_adjkernel = kernel_space.element(adjkernel)

# Discretization parameters
n = 20
npoints = np.array([n+1, n+1])
npoints_kernel = np.array([2*n+1, 2*n+1])

# Discretization spaces
disc_space = odl.uniform_discr_fromspace(cont_space, npoints)
disc_kernel_space = odl.uniform_discr_fromspace(kernel_space, npoints_kernel)

# Discretize the functions
disc_kernel = disc_kernel_space.element(cont_kernel)
disc_adjkernel = disc_kernel_space.element(cont_adjkernel)

# Create operator
conv = Convolution(disc_space, disc_kernel, disc_adjkernel)

odl.diagnostics.OperatorTest(conv).run_tests()
