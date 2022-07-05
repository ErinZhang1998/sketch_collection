"""
B-splines utilities. For reference material on B-splines, see Kristin Branson's
"A Practical Review of Uniform B-splines":
http://vision.ucsd.edu/~kbranson/research/bsplines/bsplines.pdf
"""
import functools
import torch
from torch import Tensor
import numpy as np 
# from .parameters import Parameters
# from .util.stroke import dist_along_traj

__all__ = ['vectorized_bspline_coeff', 'bspline_gen_s', 'coefficient_mat',
           'get_stk_from_bspline', 'fit_bspline_to_traj']

class Parameters:
    def __init__(self):
        # Library to use
        self.libname = 'library'
        self.set_rendering_params()
        self.set_spline_params()
        self.set_image_model_params()
        self.set_mcmc_params()
        self.set_search_params()


    def set_rendering_params(self):
        # image size
        self.imsize = torch.Size([105, 105])

        ## ink-add parameters
        # amount of ink per point
        self.ink_pp = 2.
        # distance between points to which you get full ink
        self.ink_max_dist = 2.

        ## broadening parameters
        # number of convolutions
        self.ink_ncon = 2
        # parameter 1
        self.ink_a = 0.5
        # parameter 2
        self.ink_b = 6.
        # broadening version (must be either "Lake" or "Hinton")
        self.broaden_mode = 'Lake'

        ## blurring parameters
        # convolution size for blurring
        self.fsize = 11

    def set_spline_params(self):
        """
        Parameters for creating a trajectory from a spline
        """
        # maxmium number of evaluations
        self.spline_max_neval = 200
        # minimum number of evaluations
        self.spline_min_neval = 10
        # 1 trajectory point for every this many units pixel distance
        self.spline_grain = 1.5

    def set_image_model_params(self):
        """
        Max/min noise parameters for image model
        """
        # min/max blur sigma
        self.max_blur_sigma = torch.tensor(16, dtype=torch.float)
        self.min_blur_sigma = torch.tensor(0.5, dtype=torch.float)
        # min/max pixel epsilon
        self.max_epsilon = torch.tensor(0.5, dtype=torch.float)
        self.min_epsilon = torch.tensor(1e-4, dtype=torch.float)

    def set_mcmc_params(self):
        """
        MCMC parameters
        """
        ## chain parameters
        # number of samples to take in the MCMC chain (for classif.)
        self.mcmc_nsamp_type_chain = 200
        # number of samples to store from this chain (for classif.)
        self.mcmc_nsamp_type_store = 10
        # for completion (we take last sample in this chain)
        self.mcmc_nsamp_token_chain = 25

        ## mcmc proposal parameters
        # global position move
        self.mcmc_prop_gpos_sd = 1
        # shape move
        self.mcmc_prop_shape_sd = 3/2
        # scale move
        self.mcmc_prop_scale_sd = 0.0235
        # attach relation move
        self.mcmc_prop_relmid_sd = 0.2168
        # multiply the sd of the standard position noise by this to propose
        # new positions from prior
        self.mcmc_prop_relpos_mlty = 2

    def set_search_params(self):
        """
        Parameters of search algorithm (part of inference)
        """
        # number of particles to use in search algorithm
        self.K = 5
        # scale changes must be less than a factor of 2
        self.max_affine_scale_change = 2
        # shift changes must less than this
        self.max_affine_shift_change = 50
    


PM = Parameters()

def dist_along_traj(stk):
    """
    Compute the total euclidean dist. along a stroke (or sub-stroke) trajectory

    Parameters
    ----------
    stk : torch.Tensor | np.ndarray
        (n,2) stroke trajectory

    Returns
    -------
    total_dist : float
        distance along the stroke trajectory

    """
    assert stk.shape[1] == 2
    if isinstance(stk, np.ndarray):
        distances = np.linalg.norm(stk[1:] - stk[:-1], axis=1)
    elif isinstance(stk, torch.Tensor):
        distances = torch.norm(stk[1:] - stk[:-1], dim=1)
    else:
        raise Exception
    dist = distances.sum()
    return dist

def vectorized_bspline_coeff(vi, vs):
    """Spline coefficients

    from Kristin Branson's "A Practical Review of Uniform B-splines"

    Inputs vi and vs are the spline evaluation indices and times (respectively),
    each with shape [neval, nland]. The output matrix has shape [neval,nland].
    """
    assert vi.shape == vs.shape
    assert vi.dtype == vs.dtype

    def poly(x, expn, wt):
        expn = torch.tensor(expn, dtype=C.dtype, device=C.device)
        wt = torch.tensor(wt, dtype=C.dtype, device=C.device)
        return x.unsqueeze(-1).pow(expn) @ wt

    C = torch.zeros_like(vi)

    # sel1
    sel = vs.ge(vi) & vs.lt(vi+1)
    diff = vs[sel] - vi[sel]
    C[sel] = diff.pow(3)
    # sel2
    sel = vs.ge(vi+1) & vs.lt(vi+2)
    diff = vs[sel] - vi[sel] - 1
    C[sel] = poly(diff, expn=(3,2,1,0), wt=(-3,3,3,1))
    # sel3
    sel = vs.ge(vi+2) & vs.lt(vi+3)
    diff = vs[sel] - vi[sel] - 2
    C[sel] = poly(diff, expn=(3,2,0), wt=(3,-6,4))
    # sel4
    sel = vs.ge(vi+3) & vs.lt(vi+4)
    diff = vs[sel] - vi[sel] - 3
    C[sel] = (1 - diff).pow(3)

    return C.div_(6.)


@functools.lru_cache(maxsize=128)
def bspline_gen_s(nland, neval=200, device=None):
    """Generate time points for evaluating spline.

    The convex-combination of the endpoints with five control points are 80
    percent of the last cpt and 20 percent of the control point after that.
    We return the upper and lower bounds, in addition to the timepoints.
    """
    lb = float(2)
    ub = float(nland + 1)
    s = torch.linspace(lb, ub, neval, device=device)

    return s, lb, ub


@functools.lru_cache(maxsize=128)
def coefficient_mat(nland, neval=None, s=None, device=None):
    """Generate the B-spline coefficient matrix"""

    # generate time vector
    if s is None:
        assert neval is not None, 'neval must be provided when s not provided.'
        s, _, _ = bspline_gen_s(nland, neval, device=device)
    else:
        if s.dim() == 0:
            s = s.view(1)
        assert s.dim() == 1

    # generate index vector
    i = torch.arange(nland, dtype=s.dtype, device=device)

    # generate coefficient matrix and normalize
    vs, vi = torch.meshgrid(s, i, indexing='ij')  # (neval, nland)
    C = vectorized_bspline_coeff(vi, vs)  # (neval, nland)
    C = C / C.sum(1, keepdim=True)

    return C




# ---------------------------------------------------
#    Core functions for spline fitting/evaluation
# ---------------------------------------------------

def _check_input(x):
    assert torch.is_tensor(x)
    assert x.dim() == 2
    assert x.size(1) == 2


def get_stk_from_bspline(Y, neval=None, s=None):
    """Produce a stroke trajectory by evaluating a B-spline.

    Parameters
    ----------
    Y : Tensor
        [nland,2] input spline (control points)
    neval : int
        number of eval points (optional)
    s : Tensor
        (optional) [neval] time points for spline evaluation

    Returns
    -------
    X : Tensor
        [neval,2] output trajectory
    """
    _check_input(Y)
    nland = Y.size(0)

    # if `neval` is None, set it adaptively according to stroke size
    if neval is None and s is None:
        X = get_stk_from_bspline(Y, neval=PM.spline_max_neval)
        dist = dist_along_traj(X)
        neval = (dist / PM.spline_grain).ceil().long()
        neval = neval.clamp(PM.spline_min_neval, PM.spline_max_neval).item()

    C = coefficient_mat(nland, neval, s=s, device=Y.device)
    X = torch.matmul(C, Y)  # (neval,2)

    return X


def fit_bspline_to_traj(X, nland, s=None, include_resid=False):
    """Produce a B-spline from a trajectory with least-squares.

    Parameters
    ----------
    X : Tensor
        [neval,2] input trajectory
    nland : int
        number of landmarks (control points)
    s : Tensor
        (optional) [neval] time points for spline evaluation
    include_resid : bool
        whether to return the residuals of the least-squares problem

    Returns
    -------
    Y : Tensor
        [neval,2] output spline
    residuals : Tensor
        [2,] residuals of the least-squares problem (optional)
    """
    _check_input(X)
    neval = X.size(0)

    C = coefficient_mat(nland, neval, s=s, device=X.device)
    Y, residuals, _, _ = torch.linalg.lstsq(C, X, driver='gels')

    if include_resid:
        return Y, residuals

    return Y

def fit_minimal_spline(stroke, thresh, max_nland=100, normalize=True):
    assert isinstance(stroke, torch.Tensor)
    ntraj = stroke.size(0)

    # determine num control points
    for nland in range(1, min(ntraj+1, max_nland)):
        spline, residuals = fit_bspline_to_traj(stroke, nland, include_resid=True)
        loss = torch.sum(residuals).item()
        if normalize:
            loss = loss/float(ntraj)
        if loss < thresh:
            return spline

    return spline