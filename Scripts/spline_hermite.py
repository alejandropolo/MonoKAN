import torch
# Hermite Basis Functions
def hermite_basis_old(x, x0, x1):
    """
    Calculate the Hermite basis functions for a given value x.

    The Hermite basis functions are used in interpolation and approximation
    theory. They provide a way to interpolate between two points, x0 and x1,
    with not only the function values at these points but also the derivatives
    considered. This function returns the four Hermite basis functions, which
    are essential for cubic Hermite spline interpolation.

    Parameters:
    - x (float): The point at which to evaluate the Hermite basis functions.
    - x0 (float): The starting point of the interval.
    - x1 (float): The ending point of the interval.

    Returns:
    - h00 (float): The first Hermite basis function value at x.
    - h10 (float): The second Hermite basis function value at x, related to the derivative at x0.
    - h01 (float): The third Hermite basis function value at x, related to the function value at x1.
    - h11 (float): The fourth Hermite basis function value at x, related to the derivative at x1.

    The basis functions are defined as follows:
    - h00: Controls the function value at x0.
    - h10: Controls the derivative at x0.
    - h01: Controls the function value at x1.
    - h11: Controls the derivative at x1.

    These basis functions ensure that the spline passes through the end points
    (x0, x1) with specified derivatives at these points, providing a smooth
    transition between segments of the spline.
    """
    h = x1 - x0  # Interval length
    t = (x - x0) / h  # Normalized position within the interval
    # Hermite basis functions
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)
    return h00, h10, h01, h11

def H_batch_old(x, grid, coef, device='cpu'):
    x = x.to(device)
    grid = grid.to(device)
    coef = coef.to(device)

    num_splines, _ = x.shape
    num_intervals = grid.shape[1] - 1

    y_eval = torch.zeros_like(x)
    
    for spline_idx in range(num_splines):
        try: 
            for i in range(num_intervals):
                x0 = grid[spline_idx, i]
                x1 = grid[spline_idx, i + 1]
                mask = (x[spline_idx] >= x0) & (x[spline_idx] <= x1)
                h00, h10, h01, h11 = hermite_basis(x[spline_idx][mask], x0, x1)

                y0 = coef[spline_idx, i, 0]
                dy0 = coef[spline_idx, i, 1]
                y1 = coef[spline_idx, i + 1, 0]
                dy1 = coef[spline_idx, i + 1, 1]

                y_eval[spline_idx][mask] = y0 * h00 + dy0 * (x1 - x0) * h10 + y1 * h01 + dy1 * (x1 - x0) * h11
        except:
            print('ok')

        # Linear extrapolation
        slope_start = coef[spline_idx, 0, 1]
        slope_end = coef[spline_idx, -1, 1]
        mask_start = x[spline_idx] < grid[spline_idx, 0]
        mask_end = x[spline_idx] > grid[spline_idx, -1]
        y_eval[spline_idx][mask_start] = coef[spline_idx, 0, 0] + slope_start * (x[spline_idx][mask_start] - grid[spline_idx, 0])
        y_eval[spline_idx][mask_end] = coef[spline_idx, -1, 0] + slope_end * (x[spline_idx][mask_end] - grid[spline_idx, -1])

    return y_eval

def hermite_basis(t):
    """
    Calculate the Hermite basis functions for a given normalized value t.

    Parameters:
    - t (torch.Tensor): The normalized position within the interval.

    Returns:
    - h00, h10, h01, h11 (torch.Tensor): The Hermite basis function values.
    """
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00, h10, h01, h11

def H_batch(x, grid, coef, device='cpu'):
    x = x.to(device)
    grid = grid.to(device)
    coef = coef.to(device)

    num_splines, num_points = x.shape
    num_intervals = grid.shape[1] - 1

    y_eval = torch.zeros_like(x)

    for spline_idx in range(num_splines):
        x_spline = x[spline_idx].contiguous()
        grid_spline = grid[spline_idx].contiguous()
        coef_spline = coef[spline_idx].contiguous()

        # Find interval indices for each x value
        idx = torch.searchsorted(grid_spline, x_spline, right=False) - 1
        idx = torch.clamp(idx, 0, num_intervals - 1)

        x0 = grid_spline[idx]
        x1 = grid_spline[idx + 1]
        t = (x_spline - x0) / (x1 - x0)

        h00, h10, h01, h11 = hermite_basis(t)

        y0 = coef_spline[idx, 0]
        dy0 = coef_spline[idx, 1]
        y1 = coef_spline[idx + 1, 0]
        dy1 = coef_spline[idx + 1, 1]

        y_eval[spline_idx] = y0 * h00 + dy0 * (x1 - x0) * h10 + y1 * h01 + dy1 * (x1 - x0) * h11

        # Linear extrapolation for out-of-bounds values
        slope_start = coef_spline[0, 1]
        slope_end = coef_spline[-1, 1]
        mask_start = x_spline < grid_spline[0]
        mask_end = x_spline > grid_spline[-1]
        y_eval[spline_idx][mask_start] = coef_spline[0, 0] + slope_start * (x_spline[mask_start] - grid_spline[0])
        y_eval[spline_idx][mask_end] = coef_spline[-1, 0] + slope_end * (x_spline[mask_end] - grid_spline[-1])

    return y_eval




def coef2curve_hermite(x_eval, grid, coef, device="cpu"):
    '''
    converting Hermite coefficients to Hermite curves. Evaluate x on Hermite curves (summing up H_batch results over Hermite basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 3D torch.tensor)
            shape (number of splines, number of grid intervals + 1, 2). The last dimension corresponds to y and dy
        device : str
            device
        
    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+1,2))
    >>> print(x_eval.shape,grids.shape,coef.shape)
    >>> coef2curve_hermite(x_eval, grids, coef).shape
    torch.Size([5, 100])
    '''
    # Ensure compatibility between coef and x_eval types
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    
    ## TODO: ROUND TO AVOID FLOATING POINT ERRORS
    # Evaluate the Hermite curves using the Hermite basis function
    y_eval = H_batch(x_eval, grid, coef, device=device)
    
    return y_eval


def curve2coef_hermite_old(x_eval, y_eval, grid, device="cpu"):
    '''
    Converting Hermite spline curves to Hermite spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        device : str
            device

    Returns:
    --------
        coef : 3D torch.tensor
            shape (number of splines, number of grid intervals + 1, 2). The last dimension corresponds to y and dy
    '''
    num_splines, num_samples = x_eval.shape
    num_intervals = grid.shape[1] - 1

    coef = torch.zeros((num_splines, num_intervals + 1, 2), device=device)
    
    for spline_idx in range(num_splines):
        for i in range(num_intervals):
            x0 = grid[spline_idx, i]
            x1 = grid[spline_idx, i + 1]
            mask = (x_eval[spline_idx] >= x0) & (x_eval[spline_idx] <= x1)

            if mask.sum() == 0:
                continue

            x_segment = x_eval[spline_idx][mask]
            y_segment = y_eval[spline_idx][mask]
            
            h00, h10, h01, h11 = hermite_basis(x_segment, x0, x1)
            
            A = torch.stack((h00, h10 * (x1 - x0), h01, h11 * (x1 - x0)), dim=1)
            b = y_segment

            # Solve the linear system A * [y0, dy0, y1, dy1] = b
            coeffs = torch.linalg.lstsq(A, b).solution

            coef[spline_idx, i, 0] = coeffs[0]  # y0
            coef[spline_idx, i, 1] = coeffs[1]  # dy0

            if i == num_intervals - 1:
                coef[spline_idx, i + 1, 0] = coeffs[2]  # y1
                coef[spline_idx, i + 1, 1] = coeffs[3]  # dy1

    return coef

import torch

def curve2coef_hermite(x_eval, y_eval, grid, device="cpu"):
    '''
    Converting Hermite spline curves to Hermite spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        device : str
            device

    Returns:
    --------
        coef : 3D torch.tensor
            shape (number of splines, number of grid intervals + 1, 2). The last dimension corresponds to y and dy
    '''
    num_splines, num_samples = x_eval.shape
    num_intervals = grid.shape[1] - 1

    coef = torch.zeros((num_splines, num_intervals + 1, 2), device=device)
    
    for spline_idx in range(num_splines):
        for i in range(num_intervals):
            x0 = grid[spline_idx, i]
            x1 = grid[spline_idx, i + 1]
            mask = (x_eval[spline_idx] >= x0) & (x_eval[spline_idx] <= x1)

            if mask.sum() == 0:
                continue

            x_segment = x_eval[spline_idx][mask]
            y_segment = y_eval[spline_idx][mask]
            
            t = (x_segment - x0) / (x1 - x0)
            h00, h10, h01, h11 = hermite_basis(t)
            
            A = torch.stack((h00, h10 * (x1 - x0), h01, h11 * (x1 - x0)), dim=1)
            b = y_segment

            # Solve the linear system A * [y0, dy0, y1, dy1] = b
            coeffs = torch.linalg.lstsq(A, b).solution

            coef[spline_idx, i, 0] = coeffs[0]  # y0
            coef[spline_idx, i, 1] = coeffs[1]  # dy0

            if i == num_intervals - 1:
                coef[spline_idx, i + 1, 0] = coeffs[2]  # y1
                coef[spline_idx, i + 1, 1] = coeffs[3]  # dy1

    return coef
