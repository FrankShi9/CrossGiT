import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformAttention(nn.Module):
    def __init__(self):
        super(UniformAttention, self).__init__()
        self.act = torch.randint(-1,1) * torch.normal(0,1,1) # quant normal activation
        self.w = nn.Parameter(torch.randn(1)) # quant normal weight


    def forward(self, q, k, v):
        """
        前向传播函数。
        
        Args:
            q (Tensor): 查询张量。
            k (Tensor): 键张量。
            v (Tensor): 值张量。
        
        Returns:
            Tensor: 经过注意力机制处理后的输出张量。
        """
        return self.act * (w * (q + k + v))


''' Graph-Image-Text Cross-Modal Transformer '''
class CrossGiT(nn.Module):
    def __init__(self, latent_dim, out_dim, **args, **kwargs):
        super(CrossGiT, self).__init__()
        self.conv0 = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0) # TODO
        self.conv2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        self.att = UniformAttention()
        self.fc = nn.Linear(latent_dim, out_dim, bias=False) # TODO

        self.pos = nn.Parameter(torch.randn(1, 1, 1)) # postion embedding for g i t
        self.act = torch.randint(-1,1) * torch.normal(0,1,1) # quant normal activation


    def forward(self, g, i, t):
        """
        前向传播函数。
        rich info reduces the need for norm and dropout
        Args:
            g (Tensor): 图的特征张量。
            i (Tensor): 输入的image特征张量。
            t (Tensor): 时间序列的特征张量。
        
        Returns:
            Tensor: 经过注意力机制处理后的输出张量。
        """

        gc = self.act * (self.conv0(g)) * self.pos # graph node mask
        tc = self.act * (self.conv0(t)) + self.pos # text shift
        ic = self.conv3(self.conv2(self.conv1(i))) # image fft
        
        # query is graph, key is text, value is image
        fs = self.fc(self.att(gc, tc, ic)) # att as norm

        # graph residual cross modal broadcast fusion (mix of gaussian)
        return self.att(fs + gc) 
