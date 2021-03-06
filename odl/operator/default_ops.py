﻿# Copyright 2014, 2015 Jonas Adler
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

"""Default operators defined on any (reasonable) space."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# ODL imports
from odl.operator.operator import Operator
from odl.set.pspace import ProductSpace
from odl.set.space import LinearSpace, LinearSpaceVector


__all__ = ('ScalingOperator', 'ZeroOperator', 'IdentityOperator',
           'LinCombOperator', 'MultiplyOperator',
           'InnerProductOperator', 'InnerProductAdjointOperator',
           'ConstantOperator')


class ScalingOperator(Operator):

    """Operator of multiplication with a scalar."""

    def __init__(self, space, scalar):
        """Initialize a ScalingOperator instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space of elements which the operator is acting on
        scalar : field element
            An element in the field of the space that the vectors are
            scaled with
        """
        if not isinstance(space, LinearSpace):
            raise TypeError('space {!r} not a LinearSpace instance.'
                            ''.format(space))

        super().__init__(space, space, linear=True)
        self._space = space
        self._scal = space.field.element(scalar)

    def _apply(self, x, out):
        """Scale input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            input vector to be scaled
        out : ``range`` element
            Output vector to which the result is written

        Returns
        -------
        `None`

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> op = ScalingOperator(r3, 2.0)
        >>> op(vec, out)  # Returns out
        Rn(3).element([2.0, 4.0, 6.0])
        >>> out
        Rn(3).element([2.0, 4.0, 6.0])
        """
        out.lincomb(self._scal, x)

    def _call(self, x):
        """Return the scaled element.

        Parameters
        ----------
        x : domain element
            input vector to be scaled

        Returns
        -------
        scaled : ``range`` element
            The scaled vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = ScalingOperator(r3, 2.0)
        >>> op(vec)
        Rn(3).element([2.0, 4.0, 6.0])
        """
        return self._scal * x

    @property
    def inverse(self):
        """Return the inverse operator.

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = ScalingOperator(r3, 2.0)
        >>> inv = op.inverse
        >>> inv(op(vec)) == vec
        True
        >>> op(inv(vec)) == vec
        True
        """
        if self._scal == 0.0:
            raise ZeroDivisionError('scaling operator not invertible for '
                                    'scalar==0')
        return ScalingOperator(self._space, 1.0 / self._scal)

    @property
    def adjoint(self):
        """ The adjoint is given by taking the conjugate of the scalar
        """
        # TODO: optimize to self if `scal` is real
        return ScalingOperator(self._space, self._scal.conjugate())

    def __repr__(self):
        """op.__repr__() <==> repr(op)."""
        return 'ScalingOperator({!r}, {!r})'.format(self._space, self._scal)

    def __str__(self):
        """op.__str__() <==> str(op)."""
        return '{} * I'.format(self._scal)


class ZeroOperator(ScalingOperator):

    """Operator mapping each element to the zero element."""

    def __init__(self, space):
        """Initialize a ZeroOperator instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space of elements which the operator is acting on
        """
        super().__init__(space, 0)

    def __repr__(self):
        """op.__repr__() <==> repr(op)."""
        return 'ZeroOperator({!r})'.format(self._space)

    def __str__(self):
        """op.__str__() <==> str(op)."""
        return '0'


class IdentityOperator(ScalingOperator):

    """Operator mapping each element to itself."""

    def __init__(self, space):
        """Initialize an IdentityOperator instance.

        Parameters
        ----------
        space : LinearSpace
            The space of elements which the operator is acting on
        """
        super().__init__(space, 1)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'IdentityOperator({!r})'.format(self._space)

    def __str__(self):
        """Return ``str(self)``."""
        return "I"


class LinCombOperator(Operator):

    """Operator mapping two space elements to a linear combination.

    This opertor calculates:

    ``out = a*x[0] + b*x[1]``
    """

    # pylint: disable=abstract-method
    def __init__(self, space, a, b):
        """Initialize a LinCombOperator instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space of elements which the operator is acting on
        a : scalar
            Scalar to multiply x[0] with
        b : scalar
            Scalar to multiply x[1] with
        """
        domain = ProductSpace(space, space)
        super().__init__(domain, space, linear=True)
        self.a = a
        self.b = b

    def _apply(self, x, out):
        """Linearly combine the input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            An element in the operator domain (2-tuple of space
            elements) whose linear combination is calculated
        out : ```range`` element
            Vector to which the result is written

        Examples
        --------
        >>> from odl import Rn, ProductSpace
        >>> r3 = Rn(3)
        >>> r3xr3 = ProductSpace(r3, r3)
        >>> xy = r3xr3.element([[1, 2, 3], [1, 2, 3]])
        >>> z = r3.element()
        >>> op = LinCombOperator(r3, 1.0, 1.0)
        >>> op(xy, out=z)  # Returns z
        Rn(3).element([2.0, 4.0, 6.0])
        >>> z
        Rn(3).element([2.0, 4.0, 6.0])
        """
        out.lincomb(self.a, x[0], self.b, x[1])

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'LinCombOperator({!r}, {!r}, {!r})'.format(
            self.range, self.a, self.b)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}*x + {}*y".format(self.a, self.b)


class MultiplyOperator(Operator):

    """Operator multiplying two elements.

    The multiply operator calculates:

    out = x[0] * x[1]

    This is only applicable in Algebras.
    """

    # pylint: disable=abstract-method
    def __init__(self, space):
        """Initialize a MultiplyOperator instance.

        Parameters
        ----------
        space : LinearSpace
            The space of elements which the operator is acting on
        """
        domain = ProductSpace(space, space)
        super().__init__(domain, space)

    def _apply(self, x, out):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            An element in the operator domain (2-tuple of space
            elements) whose elementwise product is calculated
        out : ``range`` element
            Vector to which the result is written

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> r3xr3 = ProductSpace(r3, r3)
        >>> xy = r3xr3.element([[1, 2, 3], [1, 2, 3]])
        >>> z = r3.element()
        >>> op = MultiplyOperator(r3)
        >>> op(xy, z)
        Rn(3).element([1.0, 4.0, 9.0])
        >>> z
        Rn(3).element([1.0, 4.0, 9.0])
        """
        out.space.multiply(x[0], x[1], out)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'MultiplyOperator({!r})'.format(self.range)

    def __str__(self):
        """Return ``str(self)``."""
        return "x * y"


class InnerProductOperator(Operator):
    """Operator taking the inner product with a fixed vector.

    The multiply operator calculates:

    ``InnerProductOperator(vec)(x) == x.inner(vec)``

    This is only applicable in inner product spaces.
    """

    # pylint: disable=abstract-method
    def __init__(self, vector):
        """Initialize a InnerProductOperator instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector to take the inner product with
        """
        self.vector = vector
        super().__init__(vector.space, vector.space.field, linear=True)

    def _call(self, x):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``vector.space`` element
            An element in the space of the vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = InnerProductOperator(x)
        >>> op(r3.element([1, 2, 3]))
        14.0
        """
        return x.inner(self.vector)

    @property
    def adjoint(self):
        return InnerProductAdjointOperator(self.vector)

    @property
    def T(self):
        return self.vector

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'InnerProductOperator({!r})'.format(self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}.T".format(self.vector)


class InnerProductAdjointOperator(Operator):
    """Operator taking the scalar product with a fixed vector.

    The multiply operator calculates:

    ``InnerProductAdjointOperator(vec)(x) == x * vec``
    """

    # pylint: disable=abstract-method
    def __init__(self, vector):
        """Initialize a InnerProductOperator instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector to take the inner product with
        """
        self.vector = vector
        super().__init__(vector.space.field, vector.space, linear=True)

    def _call(self, x):
        """Multiply by the input.

        Parameters
        ----------
        x : ``vector.space.field`` element
            An element in the field of the vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = InnerProductAdjointOperator(x)
        >>> op(3.0)
        Rn(3).element([3.0, 6.0, 9.0])
        """
        return x * self.vector

    def _apply(self, x, out):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            An element in the operator domain (2-tuple of space
            elements) whose elementwise product is calculated
        out : ``range`` element
            Vector to which the result is written

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = InnerProductAdjointOperator(x)
        >>> out = r3.element()
        >>> result = op(3.0, out=out)
        >>> result
        Rn(3).element([3.0,  6.0,  9.0])
        >>> result is out
        True
        """
        out.lincomb(x, self.vector)

    @property
    def adjoint(self):
        return InnerProductOperator(self.vector)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'InnerProductAdjointOperator({!r})'.format(self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}".format(self.vector)


class ConstantOperator(Operator):

    """ Operator that always returns the same value

    ``ConstantOperator(vector)(x) <==> vector``
    """

    def __init__(self, vector, dom=None):
        """Initialize an instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector constant to be returned

        dom : `LinearSpace`, default : vector.space
            The domain of the operator.
        """
        if not isinstance(vector, LinearSpaceVector):
            raise TypeError('space {!r} not a LinearSpaceVector instance.'
                            ''.format(vector))

        if dom is None:
            dom = vector.space

        self.vector = vector
        super().__init__(dom, vector.space)

    def _call(self, x):
        """ Returns the constant vector

        Parameters
        ----------
        x : ``domain`` element
            Any element in the domain

        Returns
        -------
        vector : `LinearSpaceVector`
            The constant vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = ConstantOperator(x)
        >>> op(x)
        Rn(3).element([1.0, 2.0, 3.0])
        >>> op(r3.zero())
        Rn(3).element([1.0, 2.0, 3.0])
        >>> op(r3.element())
        Rn(3).element([1.0, 2.0, 3.0])
        """
        return self.vector.copy()

    def _apply(self, x, out):
        """ Assign out to the constant vector

        Parameters
        ----------
        x : ``domain`` element
            Any element in the domain
        out : ``range`` element
            Vector that gets assigned to the constant vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = ConstantOperator(x)
        >>> op(x, out=r3.element())
        Rn(3).element([1.0, 2.0, 3.0])
        """
        out.assign(self.vector)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ConstantOperator({!r})'.format(self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}".format(self.vector)


class ResidualOperator(Operator):

    """ Operator that returns the residual of some operator application
    with a vector

    ``ResidualOperator(op, vector)(x) <==> op(x) - vector``
    """

    def __init__(self, op, vector):
        """ Initialize an instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector constant to be returned

        dom : `LinearSpace`, default : vector.space
            The domain of the operator.
        """
        if not isinstance(op, Operator):
            raise TypeError('op {!r} not a Operator instance.'
                            ''.format(op))

        if not isinstance(vector, LinearSpaceVector):
            raise TypeError('space {!r} not a LinearSpaceVector instance.'
                            ''.format(vector))

        if vector not in op.range:
            raise TypeError('space {!r} not in op.range {!r}.'
                            ''.format(vector, op.range))

        self.op = op
        self.vector = vector
        super().__init__(op.domain, vector.space)

    def _call(self, x):
        """ Returns the constant vector

        Parameters
        ----------
        x : ``domain`` element
            Any element in the domain

        Returns
        -------
        vector : `LinearSpaceVector`
            The constant vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = IdentityOperator(r3)
        >>> res = ResidualOperator(op, vec)
        >>> x = r3.element([4, 5, 6])
        >>> res(x)
        Rn(3).element([3.0, 3.0, 3.0])
        """
        return self.op(x) - self.vector

    def _apply(self, x, out):
        """ Assign out to the constant vector

        Parameters
        ----------
        x : ``domain`` element
            Any element in the domain
        out : ``range`` element
            Vector that gets assigned to the constant vector

        Examples
        --------
        >>> from odl import Rn
        >>> r3 = Rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = IdentityOperator(r3)
        >>> res = ResidualOperator(op, vec)
        >>> x = r3.element([4, 5, 6])
        >>> res(x, out=r3.element())
        Rn(3).element([3.0, 3.0, 3.0])
        """
        self.op(x, out)
        out -= self.vector

    def derivative(self, point):
        """ The derivative of a residual is the derivative of the operator

        ``ResidualOperator(op, vec).derivative(x) <==> op.derivative(x)``

        Parameters
        ----------
        x : ``domain`` element
            Any element in the domain where the derivative should be taken
        """
        return self.op.derivative(point)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ResidualOperator({!r}, {!r})'.format(self.op, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{} - {}".format(self.op, self.vector)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
