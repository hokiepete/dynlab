import numpy as np


from dynlab.utils import force_eigenvectors2D


def test_force_eigenvectors2D():
    assert (np.array([[[0, 1]]]) == force_eigenvectors2D(np.array([[[0, 1]]]))).all()
    assert (np.array([[[0, 1]]]) == force_eigenvectors2D(np.array([[[0, -1]]]))).all()
    assert (np.array([[[1, 0]]]) == force_eigenvectors2D(np.array([[[1, 0]]]))).all()
    assert (np.array([[[1, 0]]]) == force_eigenvectors2D(np.array([[[-1, 0]]]))).all()
    assert (np.array([[[1, 1]]]) == force_eigenvectors2D(np.array([[[1, 1]]]))).all()
    assert (np.array([[[1, -1]]]) == force_eigenvectors2D(np.array([[[-1, 1]]]))).all()
    assert (np.array([[[1, -1]]]) == force_eigenvectors2D(np.array([[[1, -1]]]))).all()
    assert (np.array([[[1, 1]]]) == force_eigenvectors2D(np.array([[[-1, -1]]]))).all()
