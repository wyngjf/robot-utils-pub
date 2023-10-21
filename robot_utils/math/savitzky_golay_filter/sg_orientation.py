import numpy as np
import scipy

def hat(vec):
    """
    Take the 3- or 6-vector representing an isomorphism of so(3) or se(3) and
    writes this as element of so(3) or se(3).

    Args:
        vec: 3- or 6-vector. Isomorphism of so(3) or se(3)

    Returns: element of so(3) or se(3)

    """
    if len(vec) == 3:
        res = np.array([
            [0, -vec(3), vec(2)],
            [vec(3), 0, -vec(1)],
            [-vec(2), vec(1), 0]
        ])
    elif len(vec) == 6:
        res = np.zeros((4, 4))
        res[:3, :3] = np.array([
            [0, -vec(6), vec(5)],
            [vec(6), 0, -vec(4)],
            [-vec(5), vec(4), 0]
        ])
        res[:3, 3] = np.array([vec(1), vec(2), vec(3)]).reshape(-1, 1)
    else:
        raise NotImplementedError(f"vec of length {len(vec)} is not supported")
    return res


def dexpSO3(a):
    """
    Computes the right trivialized tangent d exp on SO(3)

    Args:
        a: 3 vector, isomorphism to element of so(3)

    Returns: diff exponential of a
    """
    phi = np.linalg.norm(a)
    a_hat = hat(a)
    beta = np.sin(phi / 2) ^ 2 / ((phi / 2) ^ 2)
    alpha = np.sin(phi) / phi
    return np.eye(3) + 0.5 * beta * a_hat + 1 / (phi ^ 2) * (1 - alpha) * a_hat * a_hat


def expSO3(a):
    """
    Computes the exponential mapping on SO(3), Rodrigues formula

    Args:
        a: 3 vector, isomorphism to element of so(3)

    Returns: element of SO(3)

    """
    phi = np.linalg.norm(a)
    if phi != 0:
        return np.eye(3) + np.sin(phi) / phi * hat(a) + (1 - np.cos(phi)) / phi ^ 2 * hat(a) * hat(a)
    else:
        return np.eye(3)


def vee(mat):
    xi1 = (mat(3, 2) - mat(2, 3)) / 2
    xi2 = (mat(1, 3) - mat(3, 1)) / 2
    xi3 = (mat(2, 1) - mat(1, 2)) / 2

    if len(mat) == 3:
        res = np.array([xi1, xi2, xi3]).reshape(-1, 1)
    elif len(mat) == 4:
        res = np.concatenate([mat[0:3, 3], np.array([xi1, xi2, xi3])]).reshape(-1, 1)
    else:
        raise NotImplementedError
    return res


def DdexpSO3(x, z):
    """
    Directional derivative of the dexp at x in the direction of z

    Args:
        x: 3-vector, element of so(3)
        z: 3-vector, element of so(3)

    Returns: resulting derivative
    """
    hatx = hat(x)
    hatz = hat(z)
    phi = np.linalg.norm(x)

    beta = np.sin(phi / 2) ^ 2 / ((phi / 2) ^ 2)
    alpha = np.sin(phi) / phi

    res = 0.5 * beta * hatz \
        + 1 / phi ^ 2 * (1 - alpha) * (hatx * hatz + hatz * hatx) \
        + 1 / phi ^ 2 * (alpha - beta) * (x.T*z)*hatx \
        + 1 / phi ^ 2 * (beta / 2-3 / phi ^ 2 * (1-alpha))*(x.T*z)*hatx*hatx
    return res


def sg_so3(R: np.ndarray, p: int, n: int, freq: int):
    """

    Args:
        R: (3, 3, N) Noisy sequence of rotation matrices, specified as a 3-by-3-by-N matrix containing N rotation matrices
        p: Polynomial order, specified as a positive integer, greater than the window size, n
        n: Window size, specified as a positive integer.
        freq: Sample frequency, specified as positive integer.

    Returns:
        R_est     :Estimated rotation matrices, specified as a 3-by-3-by-(N-(2n+1)) matrix containing the
                   estimated rotation matrices.
        omg_est   :Estimated angular velocity, specified as a 3-by-(N-(2n+1)) vector containing the estimated
                   angular velocities at each time step.
        domg_est  :Estimated angular acceleration, specified as a 3-by-(N-(2n+1)) vector containing the estimated
                   angular accelerations at each time step.
        tf        :Time vector of the filtered signal
    """
    if not isinstance(p, int) or p >= n or n < 0 or not isinstance(n, int):
        raise ValueError('The polynomial order, p, must be a positive integer greater than the window size, n.')
    if not isinstance(freq, int) or freq < 0:
        raise ValueError("The frequency, freq, should be a positive integer.")

    N = R.shape[-1]                 # Number of samples in the sequence
    dt = 1/freq                     # Time step lower sampled
    te = N*dt                       # Total length of the sequence
    ts = np.arange(0, te, dt)       # Signal time vector
    w = np.arange(-n, n+1)          # Window for Golay
    I = np.eye(3)                   # Short hand notation
    tf = ts[n+1:N-(n+1)+1]          # Time vector filtered signal

    R_est = np.zeros((3, 3, N - len(w)))
    omg_est = np.zeros((3, N - len(w)))
    domg_est = np.zeros((3, N - len(w)))
    # Savitzky-Golay
    # For each time step (where we can apply the window)
    cnt = 0
    for ii in range(n+1, N-(n+1)+1):
        # Build matrix A and vector b based on window size w
        row = 0
        for jj in range(1, len(w)+1):
            # Time difference between 0^th element and w(jj)^th element
            Dt = ts[ii + w[jj]] - ts[ii]
            # Determine row of A matrix
            Ajj = I
            for kk in range(1, p+1):
                Ajj = cat(2, Ajj, (1/kk) * Dt**kk * I)  # Concatenation based on order n
            A[row: row + len(I), :] = Ajj
            b[row: row + len(I), :] = vee(scipy.linalg.logm(R[:, :, ii + w[jj]] / R[:, :, ii]))
            row = row + len(I)  # Update to next row
        # Solve the LS problem
        rho = np.linalg.inv(A.T.dot(A)).dot(A.T) * b
        # Obtain the coefficients of rho
