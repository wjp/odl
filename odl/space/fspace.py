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

"""Spaces of functions with common domain and range."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External imports
import numpy as np

# ODL imports
from odl.operator.operator import Operator
from odl.set.domain import IntervalProd
from odl.set.sets import RealNumbers, ComplexNumbers, Set
from odl.set.space import LinearSpace, LinearSpaceVector


__all__ = ('FunctionSet', 'FunctionSetVector',
           'FunctionSpace', 'FunctionSpaceVector')


class FunctionSet(Set):
    """A general set of functions with common domain and range."""

    def __init__(self, dom, ran):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Set`
            The domain of the functions.
        ran : `Set`
            The range of the functions.
        """
        if not isinstance(dom, Set):
            raise TypeError('domain {!r} not a `Set` instance.'.format(dom))

        if not isinstance(ran, Set):
            raise TypeError('range {!r} not a `Set` instance.'.format(dom))

        self._domain = dom
        self._range = ran

    @property
    def domain(self):
        """Return domain attribute."""
        return self._domain

    @property
    def range(self):
        """Return range attribute."""
        return self._range

    def element(self, fcall=None, fapply=None):
        """Create a `FunctionSet` element.

        Parameters
        ----------
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `range` element or a
            `numpy.ndarray` of such (vectorized call).

            If fcall is a `FunctionSetVector`, it is wrapped
            as a new `FunctionSetVector`.

        fapply : `callable`, optional
            The actual instruction for in-place evaluation.
            Its first argument must be the `range` element
            or the array of such (vectorization) to which the
            result is written.

            If fapply` is a `FunctionSetVector`, it is wrapped
            as a new `FunctionSetVector`.

        *At least one of the arguments fcall and fapply must
        be provided.*

        Returns
        -------
        element : `FunctionSetVector`
            The new element created
        """
        if isinstance(fcall, self.element_type):  # no double wrapping
            return self.element(fcall._call, fcall._apply)
        elif isinstance(fapply, self.element_type):
            return self.element(fapply._call, fapply._apply)
        else:
            return self.element_type(self, fcall, fapply)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSet` with same
            `FunctionSet.domain` and `FunctionSet.range`,
            `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FunctionSet) and
                self.domain == other.domain and
                self.range == other.range)

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSetVector`
            whose `FunctionSetVector.space` attribute
            equals this space, `False` otherwise.
        """
        return (isinstance(other, FunctionSetVector) and
                self == other.space)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'FunctionSet({!r}, {!r})'.format(self.domain, self.range)

    def __str__(self):
        """Return ``str(self)``."""
        return 'FunctionSet({}, {})'.format(self.domain, self.range)

    @property
    def element_type(self):
        """ `FunctionSetVector` """
        return FunctionSetVector


class FunctionSetVector(Operator):

    """Representation of a `FunctionSet` element."""

    def __init__(self, fset, fcall=None, fapply=None):
        """Initialize a new instance.

        Parameters
        ----------
        fset : `FunctionSet`
            The set of functions this element lives in
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).
        fapply : `callable`, optional
            The actual instruction for in-place evaluation.
            Its first argument must be the
            `FunctionSet.range` element
            or the array of such (vectorization) to which the
            result is written.

        *At least one of the arguments fcall and fapply` must
        be provided.*
        """
        if not isinstance(fset, FunctionSet):
            raise TypeError('function set {!r} not a `FunctionSet` '
                            'instance.'.format(fset))

        if fcall is None and fapply is None:
            raise ValueError('call function and apply function cannot '
                             'both be `None`.')

        if fcall is not None and not callable(fcall):
            raise TypeError('call function {!r} is not callable.'
                            ''.format(fcall))

        if fapply is not None and not callable(fapply):
            raise TypeError('apply function {!r} is not callable.'
                            ''.format(fapply))

        self._space = fset
        self._call = fcall
        self._apply = fapply

        # Todo: allow users to specify linear
        super().__init__(self.space.domain, self.space.range, linear=False)

    @property
    def space(self):
        """Return space attribute."""
        return self._space

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSetVector` with
            ``other.space`` equal to this vector's space and
            the call and apply implementations of ``other`` and
            this vector are equal. `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FunctionSetVector) and
                self.space == other.space and
                self._call == other._call and
                self._apply == other._apply)

    # FIXME: this is a bad hack bypassing the operator default
    # pattern for apply and call
    def __call__(self, *x):
        """Vectorized and multi-argument out-of-place evaluation.

        Parameters
        ----------
        x1,...,xN : `object`
            Input arguments for the function evaluation.

        Returns
        -------
        out : `FunctionSet.range` element or array of elements
            Result of the function evaluation.
        """
        if x in self.domain:
            # single value list: f(0, 1, 2)
            pass
        elif x[0] in self.domain:
            # single array: f([0, 1, 2])
            pass
        else:  # Try vectorization
            if not isinstance(self.domain, IntervalProd):
                raise TypeError('vectorized evaluation only possible for '
                                '`IntervalProd` domains.')
            # Vectorization only allowed in this case

            # First case: (N, d) array of points, where d = dimension
            if (isinstance(x[0], np.ndarray) and
                    x[0].ndim == 2 and
                    x[0].shape[1] == self.domain.ndim):
                min_coords = np.min(x[0], axis=0)
                max_coords = np.max(x[0], axis=0)

            # Second case: d meshgrid type arrays
            elif (len(x) == self.domain.ndim and
                    all(isinstance(vec, np.ndarray) for vec in x)):
                min_coords = [np.min(vec) for vec in x]
                max_coords = [np.max(vec) for vec in x]

            else:
                raise TypeError('input is neither an element of the '
                                'function domain {!r} nor an array or '
                                'meshgrid-type coordinate list.'
                                ''.format(self.domain))

            if (min_coords not in self.domain or
                    max_coords not in self.domain):
                raise ValueError('input contains points outside '
                                 '`domain` {}.'.format(self.domain))

        out = self._call(*x)

        if not (out in self.range or
                (isinstance(out, np.ndarray) and
                    out.flat[0] in self.range)):
            raise TypeError('result {!r} not an element or an array of '
                            'elements of the function range {}.'
                            ''.format(out, self.range))

        return out

    def apply(self, out, *x):
        """Vectorized and multi-argument in-place evaluation.

        Parameters
        ----------
        out : `FunctionSet.range` element or array of elements
            Element(s) to which the result is written.
        inp1,...,inpN : `object`
            Input arguments for the function evaluation.

        Returns
        -------
        `None`
        """
        if not (out in self.range or
                (isinstance(out, np.ndarray) and
                    out.flat[0] in self.range)):
            raise TypeError('result {!r} not an element or an array of '
                            'elements of the function range {}.'
                            ''.format(out, self.range))

        # TODO: no checks on input so far
        return self._apply(out, *x)

    def __str__(self):
        """Return ``str(self)``"""
        if self._call is not None:
            return str(self._call)
        else:
            return str(self._apply_impl)

    def __repr__(self):
        """Return ``repr(self)``"""
        if self._call is not None:
            return '{!r}.element({!r})'.format(self.space, self._call)
        else:
            return '{!r}.element({!r})'.format(self.space, self._apply_impl)


class FunctionSpace(FunctionSet, LinearSpace):
    """A vector space of functions."""

    def __init__(self, dom, field=RealNumbers()):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Set`
            The domain of the functions.
        field : `RealNumbers` or `ComplexNumbers`
            The range of the functions.
        """
        if not isinstance(dom, Set):
            raise TypeError('domain {!r} not a Set instance.'.format(dom))

        if not (isinstance(field, (RealNumbers, ComplexNumbers))):
            raise TypeError('field {!r} not a RealNumbers or '
                            'ComplexNumbers instance.'.format(field))

        FunctionSet.__init__(self, dom, field)
        LinearSpace.__init__(self, field)

    def element(self, fcall=None, fapply=None):
        """Create a `FunctionSpace` element.

        Parameters
        ----------
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).

            If fcall is a `FunctionSetVector`, it is wrapped
            as a new `FunctionSpaceVector`.

        fapply : `callable`, optional
            The actual instruction for in-place evaluation.
            Its first argument must be the `FunctionSet.range` element
            or the array of such (vectorization) to which the
            result is written.

            If fapply is a `FunctionSetVector`, it is wrapped
            as a new `FunctionSpaceVector`.

        Returns
        -------
        element : `FunctionSpaceVector`
            The new element.
        """
        if fcall is None and fapply is None:
            return self.zero()
        else:
            return super().element(fcall, fapply)

    def _lincomb(self, a, x, b, y, out):
        """Raw linear combination of ``x`` and ``y``.

        Notes
        -----
        The additions and multiplications are implemented via a simple
        Python function, so the resulting function is probably slow.
        """
        # Store to allow aliasing
        x_old_call = x._call
        x_old_apply = x._apply
        y_old_call = y._call
        y_old_apply = y._apply

        def lincomb_call(*x):
            """Linear combination, call version."""
            # Due to vectorization, at least one call must be made to
            # ensure the correct final shape. The rest is optimized as
            # far as possible.
            if a == 0 and b != 0:
                out = y_old_call(*x)
                if b != 1:
                    out *= b
            elif b == 0:  # Contains the case a == 0
                out = x_old_call(*x)
                if a != 1:
                    out *= a
            else:
                out = x_old_call(*x)
                if a != 1:
                    out *= a
                tmp = y_old_call(*x)
                if b != 1:
                    tmp *= b
                out += tmp

            return out

        def lincomb_apply(out, *x):
            """Linear combination, apply version."""
            # TODO: allow also CudaRn-like container types
            if not isinstance(out, np.ndarray):
                raise TypeError('in-place evaluation only possible if output '
                                'is of type `numpy.ndarray`.')
            if a == 0 and b == 0:
                out *= 0
            elif a == 0 and b != 0:
                y_old_apply(out, *x)
                if b != 1:
                    out *= b
            elif b == 0 and a != 0:
                x_old_apply(out, *x)
                if a != 1:
                    out *= a
            else:
                tmp = np.empty_like(out)
                x_old_apply(out, *x)
                y_old_apply(tmp, *x)
                if a != 1:
                    out *= a
                if b != 1:
                    tmp *= b

                out += tmp

        out._call = lincomb_call
        out._apply = lincomb_apply

    def zero(self):
        """The function mapping everything to zero.

        Notes
        -----
        Since `FunctionSpace._lincomb` is slow,
        we implement this function directly.
        """
        def zero_(*_):
            """The zero function."""
            return self.field.element(0.0)
        return self.element(zero_)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSpace` with
            same `FunctionSet.domain` and `FunctionSet.range`,
            `False` otherwise.
        """
        # TODO: equality also for FunctionSet instances?
        if other is self:
            return True

        return (isinstance(other, FunctionSpace) and
                self.domain == other.domain and
                self.range == other.range)

    def _multiply(x, y):
        """Raw pointwise multiplication of two functions.

        Notes
        -----
        The multiplication is implemented with a simple Python
        function, so the resulting function object is probably slow.
        """
        x_old = x.function
        y_old = y.function

        def product(arg):
            """The actual product function."""
            return x_old(arg) * y_old(arg)
        y._function = product

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_repr = '{!r}'.format(self.domain)
        if not isinstance(self.range, RealNumbers):
            inner_repr += ', {!r}'.format(self.range)

        return 'FunctionSpace({})'.format(inner_repr)

    def __str__(self):
        """Return ``str(self)``."""
        inner_repr = '{}'.format(self.domain)
        if not isinstance(self.range, RealNumbers):
            inner_repr += ', {}'.format(self.range)

        return 'FunctionSpace({})'.format(inner_repr)

    @property
    def element_type(self):
        """ `FunctionSpaceVector` """
        return FunctionSpaceVector


class FunctionSpaceVector(FunctionSetVector, LinearSpaceVector):
    """Representation of a `FunctionSpace` element."""

    def __init__(self, fspace, fcall=None, fapply=None):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The set of functions this element lives in
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).
        fapply : `callable`, optional
            The actual instruction for in-place evaluation.
            Its first argument must be the `FunctionSet.range`
            element or the array of such (vectorization) to which the
            result is written.

        *At least one of the arguments ``fcall`` and ``fapply`` must
        be provided.*
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('function space {!r} not a `FunctionSpace` '
                            'instance.'.format(fspace))

        super().__init__(fspace, fcall, fapply)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
