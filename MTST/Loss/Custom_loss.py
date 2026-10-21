import torch
import torch.nn as nn 

def _align_bt(y_pred, y_true):
    # [B,T,1] -> [B,T] ; [B,1] -> [B]
    if y_pred.dim()==3 and y_pred.size(-1)==1: y_pred = y_pred.squeeze(-1)
    if y_true.dim()==3 and y_true.size(-1)==1: y_true = y_true.squeeze(-1)
    return y_pred, y_true

def _pearsonr_mean(y_pred, y_true, dim=1, eps=1e-8):
    yp = y_pred - y_pred.mean(dim=dim, keepdim=True)
    yt = y_true - y_true.mean(dim=dim, keepdim=True)
    num = (yp * yt).sum(dim=dim)
    den = (yp.pow(2).sum(dim=dim).sqrt() * yt.pow(2).sum(dim=dim).sqrt() + eps)
    r = num / den
    return r.mean()

class R2BoostedLoss2(nn.Module):
    """
    Loss = alpha*MSE(ŷ,y) + beta*MSE(Δŷ,Δy) - gamma*Corr(ŷ,y)
    Compatible shapes: [B,T] ou [B,T,1] (T peut valoir 1)
    """
    def __init__(self, alpha=1.0, beta=0.8, gamma=0.2, eps=1e-8):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred, y_true = _align_bt(y_pred, y_true)  # -> [B,T] ou [B]
        # Assurer [B,T]
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        # 1) MSE niveau
        loss1 = self.mse(y_pred, y_true)

        # 2) MSE sur premières différences si T>1
        if y_pred.size(1) > 1:
            dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
            dy_true = y_true[:, 1:] - y_true[:, :-1]
            loss2 = self.mse(dy_pred, dy_true)
        else:
            loss2 = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        # 3) Corrélation moyenne par batch si T>1
        if y_pred.size(1) > 1:
            corr = _pearsonr_mean(y_pred, y_true, dim=1, eps=self.eps)
        else:
            corr = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        return self.alpha * loss1 + self.beta * loss2 - self.gamma * corr