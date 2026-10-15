import math
import warnings
from contextlib import nullcontext as _nullcontext
from functools import partial
from typing import Optional

import numpy as np
import ot as pot
import torch

class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation.

    Parameters (新增 / 变更)
    ----------
    method : str
        OT solver: "exact", "sinkhorn", "unbalanced", "partial".
    reg : float
        Regularization (ε) for Sinkhorn-family solvers.  When ``adaptive_reg=True``
        (推荐用于 latent-space CFM)，实际使用的 reg 会被乘以当前 batch cost
        矩阵的标准差，使 ε 自动适配不同尺度的 latent 空间。
    reg_m : float
        Unbalanced Sinkhorn 的 marginal regularization weight.
    normalize_cost : bool
        将 cost 矩阵归一化到 [0, 1]。  配合 ``adaptive_reg`` 时建议关闭。
    warn : bool
        收敛失败时是否打印 warning。
    numItermax : int
        Sinkhorn-family solver 最大迭代次数。  默认 1000（配合 stopThr 足够
        收敛；过大会在 underflow 后仍空转触发 numerical-error 警告）。
    stopThr : float
        Sinkhorn 收敛阈值（marginal violation < thr 即停），默认 1e-9。
    adaptive_reg : bool
        若为 True，在 ``get_map`` 中令 ``reg_eff = reg * M.std()``，使 ε 随
        batch cost 尺度自动缩放（适合未归一化的 RAE latent 空间）。
    """

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        warn: bool = True,
        numItermax: int = 1000,
        stopThr: float = 1e-9,
        adaptive_reg: bool = False,
    ) -> None:
        supported = {"exact", "sinkhorn", "unbalanced", "partial"}
        if method not in supported:
            raise ValueError(f"Unknown method: {method}, must be one of {supported}")
        self.method = method
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.adaptive_reg = adaptive_reg

    # ──────────────────────────────────────────────
    # 内部：统一 OT 求解（含自适应 reg）
    # ──────────────────────────────────────────────
    def _solve_ot(self, a, b, M_np):
        """根据 ``self.method`` 求解 OT plan，返回 plan 矩阵 ``p``。

        Parameters
        ----------
        a, b : np.ndarray
            Source / target marginal（均匀分布）。
        M_np : np.ndarray
            Cost matrix (float64)。
        """
        if self.method == "exact":
            return pot.emd(a, b, M_np)

        # ── Sinkhorn-family：计算自适应 reg ──
        reg_eff = self.reg
        if self.adaptive_reg:
            m_std = float(np.std(M_np))
            reg_eff = self.reg * (m_std + 1e-8)

        # ── warn=False 时压制 POT 的 UserWarning（numerical errors / non-converge）──
        ctx = warnings.catch_warnings() if not self.warn else _nullcontext()
        with ctx:
            if not self.warn:
                warnings.simplefilter('ignore', UserWarning)

            if self.method == "sinkhorn":
                # ── 使用 log-domain 稳定版：从根本上避免 K=exp(-M/reg) underflow ──
                # pot.sinkhorn(..., method='sinkhorn_log') 内部调用 pot.bregman.sinkhorn_log
                return pot.sinkhorn(
                    a, b, M_np, reg_eff,
                    method='sinkhorn_log',
                    numItermax=self.numItermax,
                    stopThr=self.stopThr,
                )
            elif self.method == "unbalanced":
                return pot.unbalanced.sinkhorn_knopp_unbalanced(
                    a, b, M_np, reg_eff, reg_m=self.reg_m,
                    numItermax=self.numItermax,
                    stopThr=self.stopThr,
                )
            elif self.method == "partial":
                return pot.partial.entropic_partial_wasserstein(
                    a, b, M_np, reg=reg_eff,
                    numItermax=self.numItermax,
                    stopThr=self.stopThr,
                )
            else:  # pragma: no cover  —已在 __init__ 校验
                raise ValueError(f"Unknown method: {self.method}")

    def get_map(
        self,
        x0,
        x1,
        intra_class_suboptimal: bool = False,
        suboptimal_eps: float = 1e-6,
        *,
        exclude_self_pairing: Optional[bool] = None,
        self_pairing_eps: Optional[float] = None,
    ):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            Source minibatch.
        x1 : Tensor, shape (bs, *dim)
            Target minibatch.
        intra_class_suboptimal : bool, optional
            若为 True，屏蔽 (near-)identical 配对，使 OT 退化为次优匹配。
            **仅用于类内迁移 (c→c)**；跨类迁移 (c_i→c_j) 应设为 False 以获得
            标准最优传输。

            实现方式：
              1. 屏蔽 cost ≤ ``suboptimal_eps`` 的所有 cell（同类不同 batch 中的近邻 / 重复样本）；
              2. 当 M 为方阵时额外屏蔽对角线（同 batch 恒等映射）。
            被屏蔽的 cell 会被设为一个较大的 cost（非无穷），使得 exact OT 硬性绕开，
            Sinkhorn 软性绕开（取决于 ``reg`` 大小）。
        suboptimal_eps : float, optional
            Squared-Euclidean cost 阈值，低于此值视为 identical。仅当
            ``intra_class_suboptimal=True`` 时生效。

        Deprecated aliases (向后兼容)
        -------------------------------
        exclude_self_pairing : bool | None
            ``intra_class_suboptimal`` 的旧名，若非 None 则覆盖新参数。
        self_pairing_eps : float | None
            ``suboptimal_eps`` 的旧名，若非 None 则覆盖新参数。

        Returns
        -------
        p : numpy array, shape (bs, bs)
            OT plan。
        """
        # ── 向后兼容：旧参数名覆盖 ──
        if exclude_self_pairing is not None:
            intra_class_suboptimal = exclude_self_pairing
        if self_pairing_eps is not None:
            suboptimal_eps = self_pairing_eps

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            m_max = M.max()
            if m_max > 0:
                M = M / m_max

        # ── 保存原始 cost（用于不可行时重试）──
        M_original = M.clone()

        # ── 类内次优：屏蔽近邻 / 对角线 ──
        if intra_class_suboptimal:
            M = self._apply_suboptimal_mask(M, suboptimal_eps)

        p = self._solve_ot(a, b, M.detach().cpu().numpy())

        # ── 可行性检查 + 重试 ──
        if not np.all(np.isfinite(p)) or np.abs(p.sum()) < 1e-8:
            # 可能在过度屏蔽下不可行（exact OT），放宽重试
            if intra_class_suboptimal:
                if self.warn:
                    warnings.warn(
                        "OT plan infeasible with suboptimal mask "
                        "(all cells masked?). Retrying with relaxed mask "
                        "(diagonal-only).",
                        RuntimeWarning,
                    )
                # 放宽：只屏蔽对角线（如果是方阵），不屏蔽近邻
                M_retry = M_original.clone()
                if M_retry.shape[0] == M_retry.shape[1]:
                    big_val = self._big_cost_value(M_retry)
                    diag_mask = torch.eye(
                        M_retry.shape[0], dtype=torch.bool, device=M_retry.device,
                    )
                    M_retry = M_retry.masked_fill(diag_mask, big_val)
                p = self._solve_ot(a, b, M_retry.detach().cpu().numpy())

            # 仍然不可行 → uniform fallback
            if not np.all(np.isfinite(p)) or np.abs(p.sum()) < 1e-8:
                if self.warn:
                    warnings.warn(
                        "Numerical errors in OT plan, reverting to uniform plan.",
                        RuntimeWarning,
                    )
                p = np.ones_like(p) / p.size
        return p

    # ──────────────────────────────────────────────
    # 内部：构造次优屏蔽 + 大代价值
    # ──────────────────────────────────────────────
    @staticmethod
    def _big_cost_value(M: torch.Tensor) -> float:
        """计算用于屏蔽的大代价值。

        Sinkhorn-friendly：使用 ``M.max() * 10 + M.mean()`` 而非 ``*1e3``，
        避免 entropic OT 在 log-domain 计算 overflow，同时仍足够大以使 exact OT 绕开。
        """
        m_max = M.max().item()
        m_mean = M.mean().item()
        return m_max * 10.0 + m_mean + 1.0

    def _apply_suboptimal_mask(
        self, M: torch.Tensor, suboptimal_eps: float,
    ) -> torch.Tensor:
        """在 cost 矩阵上屏蔽近邻 cell + 对角线（方阵时）。

        返回屏蔽后的 M（新 tensor）。
        """
        # 屏蔽所有 cost ≤ eps 的 cell（近邻 / 重复样本，不限位置）
        mask = (M <= suboptimal_eps)
        # 额外屏蔽对角线（同 batch 恒等）
        if M.shape[0] == M.shape[1]:
            mask = mask.clone()
            mask.fill_diagonal_(True)
        big_value = self._big_cost_value(M)
        return M.masked_fill(mask, big_value)

    def sample_map(self, pi, batch_size, replace=False):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(
        self,
        x0,
        x1,
        replace=False,
        intra_class_suboptimal: bool = False,
        suboptimal_eps: float = 1e-6,
        *,
        exclude_self_pairing: Optional[bool] = None,
        self_pairing_eps: Optional[float] = None,
    ):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            Source minibatch.
        x1 : Tensor, shape (bs, *dim)
            Target minibatch.
        replace : bool
            Sampling with/without replacement from the OT plan.
        intra_class_suboptimal : bool
            若为 True，启用类内次优匹配（屏蔽近邻/对角线）。
        suboptimal_eps : float
            屏蔽阈值（squared-Euclidean cost）。

        Deprecated aliases: ``exclude_self_pairing``, ``self_pairing_eps``
        """
        if exclude_self_pairing is not None:
            intra_class_suboptimal = exclude_self_pairing
        if self_pairing_eps is not None:
            suboptimal_eps = self_pairing_eps

        pi = self.get_map(
            x0, x1,
            intra_class_suboptimal=intra_class_suboptimal,
            suboptimal_eps=suboptimal_eps,
        )
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_index(
        self,
        x0,
        x1,
        replace=False,
        intra_class_suboptimal: bool = False,
        suboptimal_eps: float = 1e-6,
        *,
        exclude_self_pairing: Optional[bool] = None,
        self_pairing_eps: Optional[float] = None,
    ):
        r"""Compute the OT plan $\pi$ and draw samples + return indices.

        Returns ``(x0[i], x1[j], i, j)``.
        """
        if exclude_self_pairing is not None:
            intra_class_suboptimal = exclude_self_pairing
        if self_pairing_eps is not None:
            suboptimal_eps = self_pairing_eps

        pi = self.get_map(
            x0, x1,
            intra_class_suboptimal=intra_class_suboptimal,
            suboptimal_eps=suboptimal_eps,
        )
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j], i, j

    def sample_plan_with_labels(
        self,
        x0,
        x1,
        y0=None,
        y1=None,
        replace=False,
        intra_class_suboptimal: bool = False,
        suboptimal_eps: float = 1e-6,
        *,
        exclude_self_pairing: Optional[bool] = None,
        self_pairing_eps: Optional[float] = None,
    ):
        r"""Compute the OT plan $\pi$ and draw labeled samples.

        Returns ``(x0[i], x1[j], y0[i], y1[j])``.
        """
        if exclude_self_pairing is not None:
            intra_class_suboptimal = exclude_self_pairing
        if self_pairing_eps is not None:
            suboptimal_eps = self_pairing_eps

        pi = self.get_map(
            x0, x1,
            intra_class_suboptimal=intra_class_suboptimal,
            suboptimal_eps=suboptimal_eps,
        )
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret