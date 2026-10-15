
import torch

from conditional_flow_matcher.optimal_transport import OTPlanSampler

class ConditionalFlowMatcher:
    
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma
    
    @staticmethod
    def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        if isinstance(t, (float, int)):
            return t
    
        return t.reshape(-1, *([1] * (x.dim() - 1)))

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def get_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, rectified=False) -> torch.Tensor:
        if rectified:
            return x0 + t * (x1 - x0)
        else:
            return (1 - t) * x0 + t * x1
    
    def get_sigma_t(self) -> torch.Tensor:
        return self.sigma
    
    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor, rectified=False) -> torch.Tensor:

        sigma_t = self.get_sigma_t()
        sigma_t = self.pad_t_like_x(sigma_t, x0)
        mu_t = self.get_mu_t(x0, x1, t, rectified)

        return mu_t + sigma_t * epsilon

    def get_conditional_vector_field(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        returns conditional vector field ut(x1|x0) = x1 - x0
        """
        return x1 - x0
    
    def get_sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor=None,
                                                 rectified=False, mode=0) -> torch.Tensor:
        """
        mode:
            0: Normal。
            1: sigma != 0
            2: mean_flow
        """
        t = t if t is not None else torch.rand(x0.shape[0]).type_as(x0)
        t = self.pad_t_like_x(t, x0)
        eps = self.sample_noise_like(x0)

        xt = self.sample_xt(x0, x1, t, eps, rectified)
        ut = self.get_conditional_vector_field(x0, x1, t)
        # ut = self.get_conditional_vector_field(xt, x1, t)

        return t, xt, ut

    # 定义参考概率路径的函数：给定目标x1、时间t，从条件分布采样x_t，并计算条件速度场u(t, x_t|x1)
    def sample_conditional_path(self, x0, x1, t, sigma_min=0.0):
        """ 当  sigma_min=0.0 时，和 get_sample_location_and_conditional_flow 是等价的。
        根据线性插值路径采样给定时间t下的样本x_t和真实速度u(t, x_t|x1)。
        x1: [batch, 2] 目标样本
        t:  [batch] 时间 (0~1之间)
        """
        # 线性插值均值和标准差
        t = t.view(-1, 1)  # [batch, 1]
        mu_t = t * x1  # [batch, 2]
        sigma_t = (1 - t) + t * sigma_min  # [batch, 1], 广播到2维
        # 从标准正态采样一个噪声 eps
        eps = torch.randn_like(x1) if x0 is None else x0
        # 构造条件路径的样本 x_t = mu_t + sigma_t * eps
        x_t = mu_t + sigma_t * eps
        # 计算条件速度场 u(t, x_t | x1)
        # 按公式 u = (dot(sigma)/sigma) * (x - mu) + dot(mu)
        # dot(mu_t) = x1 (因为 mu_t = t*x1, 导数为 x1)
        # dot(sigma_t) = -1 + sigma_min (因为 sigma_t = 1 - (1-sigma_min)*t )
        dot_mu_t = x1
        dot_sigma_t = -1.0 + sigma_min
        # (dot(sigma)/sigma) * (x - mu)
        term1 = (dot_sigma_t / sigma_t) * (x_t - mu_t)  # [batch, 2]
        term2 = dot_mu_t  # [batch, 2]
        u_true = term1 + term2
        return x_t, u_true

    def mean_flow_sample_conditional_path(self):
        ### CVPR: Mean_Flow kaiming.
        """
        t,r = sample_t_r()
        z=(1-t)*x0+t*x1
        v= x1 - x0
        u,dudt=jvp(fn,(z,r,t),(v,0,1))
        u_tgt = v - (t-r) * dudt
        return u, stopgrad(u_tgt)
        """
        pass

