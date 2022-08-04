"""Toy code implementing the density-matrix renormalization group (DMRG)."""

import numpy as np
from a_mps import split_truncate_theta
import scipy.sparse
import scipy.sparse.linalg.eigen.arpack as arp


class HEffective(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian.

    To be diagonalized in `DMRGEngine.update_bond`. Looks like this::

        .--vL*           vR*--.
        |       i*    j*      |
        |       |     |       |
        (LP)---(W1)--(W2)----(RP)
        |       |     |       |
        |       i     j       |
        .--vL             vR--.
    """

    def __init__(self, LP, RP, W1, W2):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W1 = W1  # wL wC i i*
        self.W2 = W2  # wC wR j j*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d1, d2 = W1.shape[2], W2.shape[2]
        self.theta_shape = (chi1, d1, d2, chi2)  # vL i j vR
        self.shape = (chi1 * d1 * d2 * chi2, chi1 * d1 * d2 * chi2)
        self.dtype = W1.dtype

    def _matvec(self, theta):
        """calculate |theta'> = H_eff |theta>"""
        x = np.reshape(theta, self.theta_shape)  # vL i j vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i j vR
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))  # vL [wL*] [i] j vR, [wL] wC i [i*]
        x = np.tensordot(x, self.W2, axes=([3, 1], [0, 3]))  # vL [j] vR [wC] i, [wC] wR j [j*]
        x = np.tensordot(x, self.RP, axes=([1, 3], [0, 1]))  # vL [vR] i [wR] j, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x


class DMRGEngine(object):
    """DMRG algorithm, implemented as class holding the necessary data.

    Parameters
    ----------
    psi, model, chi_max, eps:
        See attributes

    Attributes
    ----------
    psi : MPS
        The current ground-state (approximation).
    model :
        The model of which the groundstate is to be calculated.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.split_truncate_theta`.
    LPs, RPs : list of np.Array[ndim=3]
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vL wL* vL*``, ``RPS[i]`` has legs ``vR* wR* vR``
    """

    def __init__(self, psi, model, chi_max=100, eps=1.e-12):
        assert psi.L == model.L  # ensure compatibility
        self.H_mpo = model.H_mpo #shouldn't be init_H_mpo? 
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        # initialize left and right environment
        D = self.H_mpo[0].shape[0]
        chi = psi.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype="float")  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype="float")  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i)

    def sweep(self):
        # sweep from left to right
        for i in range(self.psi.L - 2):
            self.update_bond(i)
        # sweep from right to left
        for i in range(self.psi.L - 2, 0, -1):
            self.update_bond(i)

    def update_bond(self, i):
        j = i + 1
        # get effective Hamiltonian
        Heff = HEffective(self.LPs[i], self.RPs[j], self.H_mpo[i], self.H_mpo[j])
        # Diagonalize Heff, find ground state `theta`
        theta0 = np.reshape(self.psi.get_theta2(i), [Heff.shape[0]])  # initial guess
        e, v = arp.eigsh(Heff, k=1, which='SA', return_eigenvectors=True, v0=theta0)
        theta = np.reshape(v[:, 0], Heff.theta_shape)
        # split and truncate
        Ai, Sj, Bj = split_truncate_theta(theta, self.chi_max, self.eps)
        # put back into MPS
        Gi = np.tensordot(np.diag(self.psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
        self.psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC*] vC
        self.psi.Ss[j] = Sj  # vC
        self.psi.Bs[j] = Bj  # vC j vR
        self.update_LP(i)
        self.update_RP(j)
        #return e

    def update_RP(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        j = i - 1
        RP = self.RPs[i]  # vR* wR* vR
        B = self.psi.Bs[i]  # vL i vR
        Bc = B.conj()  # vL* i* vR*
        W = self.H_mpo[i]  # wL wR i i*
        RP = np.tensordot(B, RP, axes=[2, 0])  # vL i [vR], [vR*] wR* vR
        RP = np.tensordot(RP, W, axes=[[1, 2], [3, 1]])  # vL [i] [wR*] vR, wL [wR] i [i*]
        RP = np.tensordot(RP, Bc, axes=[[1, 3], [2, 1]])  # vL [vR] wL [i], vL* [i*] [vR*]
        self.RPs[j] = RP  # vL wL vL* (== vR* wR* vR on site i-1)

    def update_LP(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`."""
        j = i + 1
        LP = self.LPs[i]  # vL wL vL*
        B = self.psi.Bs[i]  # vL i vR
        G = np.tensordot(B, np.diag(self.psi.Ss[j]**-1), axes=[2, 0])  # vL i [vR], [vR*] vR
        A = np.tensordot(np.diag(self.psi.Ss[i]), G, axes=[1, 0])  # vL [vL*], [vL] i vR
        Ac = A.conj()  # vL* i* vR*
        W = self.H_mpo[i]  # wL wR i i*
        LP = np.tensordot(LP, A, axes=[2, 0])  # vL wL* [vL*], [vL] i vR
        LP = np.tensordot(W, LP, axes=[[0, 3], [1, 2]])  # [wL] wR i [i*], vL [wL*] [i] vR
        LP = np.tensordot(Ac, LP, axes=[[0, 1], [2, 1]])  # [vL*] [i*] vR*, wR [i] [vL] vR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)
