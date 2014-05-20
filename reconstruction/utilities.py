import numpy as np

def complexStack(array_complex):
    """Takes a complex array A of dimension (n,m) or (n,1) and stacks it into 
    a real matrix/vector of dimension (2n, 2m) / (2n,1) in following formats. 
    
    matrix: [ [real(A), -imag(A)],
              [imag(A),  real(A)] ]

    vector: [ [real(A)], 
              [imag(A)] ]

    Parameters
    ----------
    array_complex : complex 2 dimensional array
        An error is raised if an array is given which does not have a well defined
        second dimension, i.e. array_complex.shape = (5,)

    Returns
    -------
    vec_not_complex : real valued array of shape (2n,1)
    mat_not_complex : real valued array of shape (2n, 2m)

    """

    try:
        second_index = array_complex.shape[1]
    except IndexError:
        raise IndexError( "Input array must have a defined second dimension" )

    array_real = np.real(array_complex)
    array_imag = np.imag(array_complex)

    # if array is a vector
    if second_index == 1:

        vec_not_complex = np.vstack( (array_real, array_imag) )

        return vec_not_complex       

    # if array is a matrix
    if second_index > 1:

        mat_row1 = np.hstack( (array_real, -array_imag) )
        mat_row2 = np.hstack( (array_imag,  array_real) )

        mat_not_complex = np.vstack( (mat_row1, mat_row2) )

        return mat_not_complex

def complexUnstack(array_not_complex):
    """Performs the reverse of 'complexStack'."""

    try:
        second_index = array_not_complex.shape[1]
    except IndexError:
        raise IndexError( "Input array must have a defined second dimension" )

    # if array is a vector
    if second_index == 1:

        vec_split = np.split(array_not_complex, 2)
        vec_complex = vec_split[0] + 1j * vec_split[1]

        return vec_complex      

    # if array is a matrix
    if second_index > 1:

        mat_split = np.split( array_not_complex[:, :array_not_complex.shape[1]/2], 2)

        mat_complex = mat_split[0] + 1j * mat_split[1]

        return mat_complex

def interpDispGrid(dispGrid, factor = 2):
    """Takes a displacement grid of real and imaginary values and returns a
    grid with an increased number of values between the minimum and maximum.
    """
    r_max = np.max(np.real(dispGrid))
    r_min = np.min(np.real(dispGrid))
    r_len = dispGrid.shape[1] - 1

    i_max = np.max(np.imag(dispGrid))
    i_min = np.min(np.imag(dispGrid))
    i_len = dispGrid.shape[0] - 1

    r_vec = np.linspace(r_min, r_max, r_len * factor + 1)
    i_vec = np.linspace(i_min, i_max, i_len * factor + 1)

    X,Y = np.meshgrid(r_vec, i_vec)

    dispGrid_interpolated = X + 1j * Y
    
    return dispGrid_interpolated

def extendDispGrid(dispGrid, alpha_max = 4):
    """Takes a grid of displacements and pads zeros up until a maximum set alpha."""
    """UNFINISHED"""
    r_max = np.max(np.real(dispGrid))
    r_min = np.min(np.real(dispGrid))
    r_len = dispGrid.shape[1] - 1

    i_max = np.max(np.imag(dispGrid))
    i_min = np.min(np.imag(dispGrid))
    i_len = dispGrid.shape[0] - 1



    r_vec = np.linspace(r_min, r_max, r_len * factor + 1)
    i_vec = np.linspace(i_min, i_max, i_len * factor + 1)

    X,Y = np.meshgrid(r_vec, i_vec)

    dispGrid_interpolated = X + 1j * Y
    
    return dispGrid_interpolated


