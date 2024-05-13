import torch
import torch.nn as nn
import torch.nn.functional as F


class L_exp(nn.Module):
    def __init__(self, opt):
        super(L_exp, self).__init__()
        self.loss_weight = opt["loss_weight"]
        self.pool = nn.AvgPool2d(opt["patch_size"])
        self.mean_val = opt["mean_val"]

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return self.loss_weight * torch.mean(d)
    

class L_spa(nn.Module):
    def __init__(self, loss_weight):
        super(L_spa, self).__init__()
        self.loss_weight = loss_weight
        self.kernel_left = nn.Parameter(
            torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.kernel_right = nn.Parameter(
            torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.kernel_up = nn.Parameter(
            torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.kernel_down = nn.Parameter(
            torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_pool = self.pool(torch.mean(org, 1, keepdim=True))
        enhance_pool = self.pool(torch.mean(enhance, 1, keepdim=True))

        def compute_gradient_loss(org_pool, enhance_pool, kernel):
            D_org = F.conv2d(org_pool, kernel, padding=1)
            D_enhance = F.conv2d(enhance_pool, kernel, padding=1)
            return torch.pow(D_org - D_enhance, 2)

        D_left = compute_gradient_loss(org_pool, enhance_pool, self.kernel_left)
        D_right = compute_gradient_loss(org_pool, enhance_pool, self.kernel_right)
        D_up = compute_gradient_loss(org_pool, enhance_pool, self.kernel_up)
        D_down = compute_gradient_loss(org_pool, enhance_pool, self.kernel_down)

        E = D_left + D_right + D_up + D_down
        return self.loss_weight * torch.mean(E)
    

class L_Smoothness(nn.Module):
    def __init__(self, loss_weight=1):
        super(L_Smoothness, self).__init__()

        self.loss_weight = loss_weight

    def forward(self, x):
        y_variation = F.l1_loss(x[:, :, :-1, :], x[:, :, 1:, :], reduction='mean')
        x_variation = F.l1_loss(x[:, :, :, :-1], x[:, :, :, 1:], reduction='mean')

        variation = y_variation + x_variation
        return self.loss_weight * variation
