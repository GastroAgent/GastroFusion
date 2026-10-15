import torch
import numpy as np

from conditional_flow_matcher.optimal_transport import OTPlanSampler, wasserstein
from conditional_flow_matcher.cond_flow_matcher import ConditionalFlowMatcher

class OptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    A Conditional Flow Matcher that uses Optimal Transport for more sophisticated 
    sampling and vector field computation.
    
    This class enhances the base Conditional Flow Matcher by:
    1. Using Optimal Transport to guide sampling between distributions
    2. Providing more flexible sampling strategies
    3. Incorporating Wasserstein distance computations
    """
    
    def __init__(self, 
                 sigma: float = 0.1, 
                 ot_method: str = 'unbalanced', # unbalanced, sinkhorn
                 ot_reg: float = 0.05, 
                 ot_normalize_cost: bool = False,
                 ot_num_iter_max: int = 1000,
                 ot_stop_thr: float = 1e-9,
                 ot_warn: bool = True,
                 ot_adaptive_reg: bool = False):
        """
        Initialize the Optimal Transport Conditional Flow Matcher
        
        Parameters
        ----------
        sigma : float, optional
            Noise scale parameter, by default 0.1
        ot_method : str, optional
            Optimal Transport method to use, by default 'sinkhorn'
        ot_reg : float, optional
            Regularization parameter for OT solver, by default 0.05
        ot_normalize_cost : bool, optional
            Whether to normalize cost matrix in OT solver, by default False
        ot_num_iter_max : int, optional
            Max iterations for Sinkhorn-family solvers, by default 1000.
            （配合 stopThr 足够收敛；过大会在 underflow 后仍空转触发警告。）
        ot_stop_thr : float, optional
            Sinkhorn 收敛阈值，marginal violation < thr 即停。默认 1e-9。
        ot_warn : bool, optional
            收敛失败 / 数值错误时是否打印 warning。默认 True。  设为 False 可
            压制 POT 的 "numerical errors" / "did not converge" 警告。
        ot_adaptive_reg : bool, optional
            If True, reg_eff = reg * M.std() to adapt to latent-space cost
            scale (recommended for unnormalized RAE latent). Default False.
        """
        super().__init__(sigma)
        
        # Initialize Optimal Transport Plan Sampler
        self.ot_sampler = OTPlanSampler(
            method=ot_method, 
            reg=ot_reg, 
            normalize_cost=ot_normalize_cost,
            numItermax=ot_num_iter_max,
            stopThr=ot_stop_thr,
            warn=ot_warn,
            adaptive_reg=ot_adaptive_reg,
        )

    def get_sample_location_and_conditional_flow(self, 
                                                 x0: torch.Tensor, 
                                                 x1: torch.Tensor, 
                                                 t: torch.Tensor = None,
                                                 sample_plan = False,
                                                 cond: dict = None,
                                                  rectified=False,
                                                  replace=False,
                                                  print_info=True,
                                                  intra_class_suboptimal: bool = False,
                                                  suboptimal_eps: float = 1e-6,
                                                  *,
                                                  exclude_self_pairing=None,
                                                  self_pairing_eps=None) -> tuple:
        """
        Sample location and conditional flow using Optimal Transport
        
        Parameters
        ----------
        x0 : torch.Tensor
            Source distribution
        x1 : torch.Tensor
            Target distribution
        t : torch.Tensor, optional
            Time step, if None, sampled randomly
        intra_class_suboptimal : bool, optional
            If True, the OT plan is forbidden to pair (near) identical samples
            between x0 and x1, including the diagonal when x0 and x1 come from the
            same batch.  This is the **类内迁移 (c→c)** 模式：OT 退化为次优匹配，
            避免 trivial 配对 (x0 ~= x1)。

            **类间迁移 (c_i→c_j) 应设为 False**，以获得标准最优传输。
            Default False.
        suboptimal_eps : float, optional
            Squared-Euclidean distance threshold below which two samples are
            considered identical. Only used when intra_class_suboptimal=True.

        Deprecated aliases (向后兼容)
        -------------------------------
        exclude_self_pairing, self_pairing_eps : 旧参数名，若非 None 则覆盖新参数。
        
        Returns
        -------
        tuple
            Sampled location and conditional flow
        """
        # ── 向后兼容：旧参数名覆盖 ──
        if exclude_self_pairing is not None:
            intra_class_suboptimal = exclude_self_pairing
        if self_pairing_eps is not None:
            suboptimal_eps = self_pairing_eps

        if sample_plan:
            if cond is not None and 'y0' in cond and 'y1' in cond:
                if print_info:
                    print('-----------------Sample Plan with Labels-----------------')
                x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(
                    x0, x1, cond['y0'], cond['y1'], replace=replace,
                    intra_class_suboptimal=intra_class_suboptimal,
                    suboptimal_eps=suboptimal_eps,
                )
                cond['y0'] = y0
                cond['y1'] = y1
            elif cond is not None and len(cond) > 0:
                if print_info:
                    print('-----------------Sample Plan with Indexes-----------------')
                    print('Old in OTCFM: --------------------------------------')
                    print('\n'.join(cond['caption']))
                    print('y: ', cond['y'].detach().cpu().numpy())

                x0, x1, i, j = self.ot_sampler.sample_plan_with_index(
                    x0, x1,
                    intra_class_suboptimal=intra_class_suboptimal,
                    suboptimal_eps=suboptimal_eps,
                )
                for k, v in cond.items():
                    if v is None:
                        continue
                    if k != 'x0' and k != 'x1':
                        if 'x0' in k:
                            if isinstance(v, torch.Tensor):
                                cond[k] = v[i]
                            elif isinstance(v, list):
                                captions = cond[k]
                                new_captions = []
                                for i_idx in i:
                                    new_captions.append(captions[i_idx])
                                cond[k] = new_captions
                            else:
                                captions = cond[k]
                                new_captions = []
                                for i_idx in i:
                                    new_captions.append(captions[i_idx])
                                cond[k] = new_captions
                            continue

                        if isinstance(v, torch.Tensor):
                            cond[k] = v[j]
                        elif isinstance(v, list):
                            captions = cond[k]
                            new_captions = []
                            for j_idx in j:
                                new_captions.append(captions[j_idx])
                            cond[k] = new_captions
                        else:
                            captions = cond[k]
                            new_captions = []
                            for j_idx in j:
                                new_captions.append(captions[j_idx])
                            cond[k] = new_captions

                if print_info:
                    print(f'i: {i}, j: {j}')
                    print('New in OTCFM: --------------------------------------')
                    print('\n'.join(cond['caption']))
                    print('y: ', cond['y'].detach().cpu().numpy())
            else:
                x0, x1 = self.ot_sampler.sample_plan(
                    x0, x1, replace=replace,
                    intra_class_suboptimal=intra_class_suboptimal,
                    suboptimal_eps=suboptimal_eps,
                )
        cond['x0'] = x0
        cond['x1'] = x1
        return super().get_sample_location_and_conditional_flow(x0, x1, t, rectified)


if __name__ == "__main__":
    # Basic sanity tests for the exclude_self_pairing option.
    torch.manual_seed(0)
    np.random.seed(0)

    def _make_batch(n=8, d=4, scale=1.0):
        return torch.randn(n, d) * scale

    print("=" * 70)
    print("Test 1: default behavior (exclude_self_pairing=False)")
    print("=" * 70)
    x0 = _make_batch()
    x1 = _make_batch()
    matcher = OptimalTransportConditionalFlowMatcher(sigma=0.0, ot_method="exact")
    cond = {}
    t, xt, ut = matcher.get_sample_location_and_conditional_flow(
        x0, x1, sample_plan=True, cond=cond, print_info=True
    )
    print("t shape:", tuple(t.shape))
    print("xt shape:", tuple(xt.shape))
    print("ut shape:", tuple(ut.shape))
    expected_ut = cond['x1'] - cond['x0']
    print("ut == x1 - x0 after OT pairing:", torch.allclose(ut, expected_ut))
    assert torch.allclose(ut, expected_ut), "Conditional flow should equal x1-x0"
    print("PASS\n")

    print("=" * 70)
    print("Test 2: same batch, exclude_self_pairing=True (forbid diagonal)")
    print("=" * 70)
    batch = _make_batch(n=8, d=4)
    x0 = batch.clone()
    x1 = batch.clone()  # x0 and x1 are the same batch
    matcher = OptimalTransportConditionalFlowMatcher(sigma=0.0, ot_method="exact")
    cond = {}
    t, xt, ut = matcher.get_sample_location_and_conditional_flow(
        x0, x1, sample_plan=True, cond=cond, print_info=True,
        exclude_self_pairing=True,
    )
    # Since x0 == x1, forbid self pairing should make every paired (x0_i, x1_j) differ.
    diff = (cond['x0'] - cond['x1']).abs().sum(dim=1)
    print("per-sample |x0 - x1| (should all be > 0):", diff.tolist())
    assert (diff > 1e-6).all(), "No paired sample should be identical when exclude_self_pairing=True"
    print("PASS\n")

    print("=" * 70)
    print("Test 3: same class, different batch, with off-diagonal duplicate")
    print("=" * 70)
    # Two different batches of the same class; inject a duplicate of x0[0] into x1[3].
    x0 = _make_batch(n=8, d=4)
    x1 = _make_batch(n=8, d=4)
    x1[3] = x0[0].clone()  # off-diagonal duplicate
    matcher = OptimalTransportConditionalFlowMatcher(sigma=0.0, ot_method="exact")

    # First, without exclusion: OT may pair x0[0] with x1[3] (distance 0).
    cond_off = {}
    matcher.get_sample_location_and_conditional_flow(
        x0, x1, sample_plan=True, cond=cond_off, print_info=True,
        exclude_self_pairing=False,
    )
    cost_off = (cond_off['x0'] - cond_off['x1']).pow(2).sum(dim=1)
    print("without exclusion, paired costs:", cost_off.tolist())
    print("min paired cost (without exclusion):", cost_off.min().item())

    # With exclusion: the near-identical pair should be avoided.
    cond_on = {}
    matcher.get_sample_location_and_conditional_flow(
        x0, x1, sample_plan=True, cond=cond_on, print_info=True,
        exclude_self_pairing=True, self_pairing_eps=1e-6,
    )
    cost_on = (cond_on['x0'] - cond_on['x1']).pow(2).sum(dim=1)
    print("with exclusion, paired costs:", cost_on.tolist())
    print("min paired cost (with exclusion):", cost_on.min().item())
    assert cost_on.min().item() > 1e-6, "Exclusion should avoid near-identical pairing"
    print("PASS\n")

    print("All tests passed.")
