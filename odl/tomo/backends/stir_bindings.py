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


"""Backend for STIR: Software for Tomographic Reconstruction

Back and forward projectors for PET.

`ForwardProjectorByBinWrapper` and `BackProjectorByBinWrapper` are general
objects of STIR projectors and back-projectors, these can be used to wrap a
given projector.

`stir_projector_from_file` allows users a easy way to create a
`ForwardProjectorByBinWrapper` by giving file paths to the required templates.

References
----------
See the `STIR webpage`_ for more information and the `STIR doc`_ for info on
the STIR classes used here.

.. _STIR webpage: http://stir.sourceforge.net
.. _STIR doc: http://stir.sourceforge.net/documentation/doxy/html/
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator.operator import Operator
from odl.discr.lp_discr import uniform_discr
from odl.space.fspace import FunctionSpace
from odl.set.domain import Cuboid

try:
    import stir
    STIR_AVAILABLE = True
except ImportError:
    STIR_AVAILABLE = False

__all__ = ('ForwardProjectorByBinWrapper',
           'BackProjectorByBinWrapper',
           'stir_projector_from_file',
           'STIR_AVAILABLE')


class StirVerbosity(object):
    """ Set STIR verbosity to a fixed level """
    def __init__(self, verbosity):
        self.verbosity = verbosity

    def __enter__(self):
        self.old_verbosity = stir.Verbosity.get()
        stir.Verbosity.set(self.verbosity)

    def __exit__(self, *_):
        stir.Verbosity.set(self.old_verbosity)


class ForwardProjectorByBinWrapper(Operator):

    """ A forward projector using STIR.

    Uses "ForwardProjectorByBinUsingProjMatrixByBin" as a projector.
    """

    def __init__(self, dom, ran, volume, proj_data,
                 projector=None, adjoint=None):
        """ Initialize a new projector.

        Parameters
        ----------
        dom : `DiscreteLp`
            Volume of the projection. Needs to have the same shape as
            ``volume.shape()``.
        ran : `DiscreteLp`
            Projection space. Needs to have the same shape as
            ``proj_data.to_array().shape()``.
        volume : ``stir.FloatVoxelsOnCartesianGrid``
            The stir volume to use in the forward projection
        proj_data : ``stir.ProjData``
            The stir description of the projection.
        projector : ``stir.ForwardProjectorByBin``, optional
            A pre-initialized projector.
        adjoint : `BackProjectorByBinWrapper`, optional
            A pre-initialized adjoint.
        """
        # Check data sizes
        if dom.shape != volume.shape():
            raise ValueError('dom.shape {} does not equal volume shape {}'
                             ''.format(dom.shape, volume.shape()))
        # Todo: improve
        proj_shape = proj_data.to_array().shape()
        if ran.shape != proj_shape:
            raise ValueError('ran.shape {} does not equal proj shape {}'
                             ''.format(ran.shape, proj_shape))

        # Set domain, range etc
        super().__init__(dom, ran, True)

        # Read template of the projection
        self.proj_data = proj_data
        self.proj_data_info = proj_data.get_proj_data_info()
        self.volume = volume

        # Create forward projection by matrix
        if projector is None:
            proj_matrix = stir.ProjMatrixByBinUsingRayTracing()
            proj_matrix.set_up(self.proj_data_info, self.volume)
            self.projector = stir.ForwardProjectorByBinUsingProjMatrixByBin(proj_matrix)
            self.projector.set_up(self.proj_data_info, self.volume)

            # If no adjoint was given, we initialize a projector here to
            # save time.
            if adjoint is None:
                back_projector = stir.BackProjectorByBinUsingProjMatrixByBin(proj_matrix)
                back_projector.set_up(self.proj_data.get_proj_data_info(), self.volume)
        else:
            # If user wants to provide both a projector and a back-projector,
            # he should wrap the back projector in an Operator
            self.projector = projector
            back_projector = None

        # Pre-create an adjoint to save time
        if adjoint is None:
            self._adjoint = BackProjectorByBinWrapper(self.range, self.domain,
                                                      self.volume, self.proj_data,
                                                      back_projector, self)
        else:
            self._adjoint = adjoint

    def _apply(self, volume, out):
        """Forward project a volume."""
        # Set volume data
        self.volume.fill(volume.asarray().flat)

        # project
        with StirVerbosity(0):
            self.projector.forward_project(self.proj_data, self.volume)

        # make odl data
        out[:] = stir.stirextra.to_numpy(self.proj_data)

    @property
    def adjoint(self):
        """The back-projector associated with this operator."""
        return self._adjoint


class BackProjectorByBinWrapper(Operator):

    """The back projector using STIR.
    """

    def __init__(self, dom, ran, volume, proj_data,
                 back_projector=None, adjoint=None,
                 method=None):
        """Initialize a new back-projector.

        Parameters
        ----------
        dom : `DiscreteLp`
            Projection space. Needs to have the same shape as
            ``proj_data.to_array().shape()``.
        ran : `DiscreteLp`
            Volume of the projection. Needs to have the same shape as
            ``volume.shape()``.
        volume : ``stir.FloatVoxelsOnCartesianGrid``
            The stir volume to use in the forward projection
        projection_data : ``stir.ProjData``
            The stir description of the projection.
        back_projector : ``stir.BackProjectorByBin``, optional
            A pre-initialized back-projector.
        adjoint : `ForwardProjectorByBinWrapper`, optional
            A pre-initialized adjoint.

        References
        ----------
        See `STIR doc`_ for info on the STIR classes.

        .. _STIR doc: http://stir.sourceforge.net/documentation/doxy/html/
        """

        # Check data sizes
        if ran.shape != volume.shape():
            raise ValueError('ran.shape {} does not equal volume shape {}'
                             ''.format(ran.shape, volume.shape()))
        # Todo: improve
        proj_shape = proj_data.to_array().shape()
        if dom.shape != proj_shape:
            raise ValueError('dom.shape {} does not equal proj shape {}'
                             ''.format(ran.shape, proj_shape))

        # Set range domain
        super().__init__(dom, ran, True)

        # Read template of the projection
        self.proj_data = proj_data
        self.proj_data_info = proj_data.get_proj_data_info()
        self.volume = volume

        # Create forward projection by matrix
        if back_projector is None:
            proj_matrix = stir.ProjMatrixByBinUsingRayTracing()
            proj_matrix.set_up(self.proj_data_info, self.volume)

            self.back_projector = stir.BackProjectorByBinUsingProjMatrixByBin(proj_matrix)
            self.back_projector.set_up(self.proj_data.get_proj_data_info(), self.volume)

            if adjoint is None:
                projector = stir.ForwardProjectorByBinUsingProjMatrixByBin(proj_matrix)
                projector.set_up(self.proj_data_info, self.volume)

        else:
            self.back_projector = back_projector
            projector = None

        # Pre-create an adjoint to save time
        if adjoint is None:
            self._adjoint = ForwardProjectorByBinWrapper(self.range, self.domain,
                                                         self.volume, self.proj_data,
                                                         projector, self)
        else:
            self._adjoint = adjoint

    def _apply(self, projections, out):
        """Back project."""
        # Set projection data
        self.proj_data.fill(projections.asarray().flat)

        # back-project
        with StirVerbosity(0):
            self.back_projector.back_project(self.volume, self.proj_data)

        # make odl data
        out[:] = stir.stirextra.to_numpy(self.volume)


def stir_projector_from_file(volume_file, projection_file):
    """ Create a STIR projector from given template files.

    Parameters
    ----------
    volume_file : `str`
        Full file path to the STIR input file containing information on the
        volume. This is usually a '.hv' file. For STIR reasons,
        a '.v' file is also needed.
    projection_file : `str`
        Full file path to the STIR input file with information on the
        projection data. This is usually a '.hs' file. For STIR reasons,
        a '.s' file is also needed.

    Returns
    -------
    projector : `ForwardProjectorByBinWrapper`
        A STIR forward projector.
    """
    volume = stir.FloatVoxelsOnCartesianGrid.read_from_file(volume_file)

    proj_data_in = stir.ProjData.read_from_file(projection_file)
    proj_data = stir.ProjDataInMemory(proj_data_in.get_exam_info(),
                                      proj_data_in.get_proj_data_info())

    origin = volume.get_origin()
    grid_spacing = volume.get_grid_spacing()
    grid_shape = [volume.get_z_size(),
                  volume.get_y_size(),
                  volume.get_x_size()]
    min_corner = [origin[1], origin[2], origin[3]]
    max_corner = [origin[1] + grid_spacing[1] * grid_shape[0],
                  origin[2] + grid_spacing[2] * grid_shape[1],
                  origin[3] + grid_spacing[3] * grid_shape[2]]

    # reverse to handle STIR bug? See:
    # https://github.com/UCL/STIR/issues/7
    recon_sp = uniform_discr(FunctionSpace(Cuboid(min_corner, max_corner)),
                             grid_shape,
                             dtype='float32')

    # TODO: set correct projection space
    proj_shape = proj_data.to_array().shape()

    data_sp = uniform_discr(FunctionSpace(Cuboid([0, 0, 0], proj_shape)),
                            proj_shape,
                            dtype='float32')

    return ForwardProjectorByBinWrapper(recon_sp, data_sp, volume, proj_data)
