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


"""ODL is a functional analysis library with a focus on discretization.

ODL suppors abstract sets, linear vector spaces defined on such
and Operators/Functionals defined on these sets. It is intended
to be used to write general code and faciliate code reuse.
"""

from __future__ import absolute_import

__version__ = '0.9b1'
__all__ = ('diagnostics', 'discr', 'operator', 'set', 'space', 'solvers',
           'trafos')


# Propagate names defined in __all__ of all submodules into the top-level
# module

from . import diagnostics

from . import discr
from .discr import *
__all__ += discr.__all__

from . import operator
from .operator import *
__all__ += operator.__all__

from . import set
from .set import *
__all__ += set.__all__

from . import space
from .space import *
__all__ += space.__all__

from . import trafos
from .trafos import *
__all__ += trafos.__all__

from . import solvers
from . import tomo
