import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# torch.manual_seed(1)


class UniformAttention(nn.Module):
    def __init__(self):
        super(UniformAttention, self).__init__()
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
        act = (torch.randint(0,2,(1,))*2-1) * torch.normal(0,1,size=(1,)) # quant normal activation
        
        k = torch.remainder(k, 2) * k
        v = torch.logical_or(torch.rand(v.shape), v) * v

        q = q.unsqueeze(-1)

        return act * (self.w * (q + k + v))


''' Graph-Image-Text Cross-Modal Transformer '''
class CrossGiT(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(CrossGiT, self).__init__()
        self.conv0 = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0) # TODO
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.att = UniformAttention()
        self.f = nn.Linear(16, out_dim, bias=False)
        self.fc = nn.Linear(latent_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(32)

        self.pos = nn.Parameter(torch.randn(1)) # postion embedding

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
        act = torch.randint(2,(1,)) * torch.normal(0,1,size=(1,)) # quant normal activation
        gc = act * self.conv0(self.conv0(g)) * self.pos # graph node mask
        tc = act * (self.conv0(t)) + self.pos # text shift
        ic = self.conv3(self.conv2(self.conv1(i))) * self.pos # image fft
        
        # query is graph, key is text, value is image
        fs = F.leaky_relu(self.f(self.att(gc, tc, ic))) # att as norm
        fs = fs.squeeze(3)
        
        fs = self.bn(fs)

        # # graph residual cross modal broadcast fusion (mix of gaussian)
        at = self.att(fs, gc+ic, ic+tc)
        at = at.permute((0, 3, 2, 1))

        return self.fc(at)


def reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        print('reset')


if __name__ == '__main__':
    model = CrossGiT(32, 1)

    graph = torch.randn(32, 1, 16)
    image = torch.randn(32, 3, 16, 16)
    text = torch.randn(32, 1, 16)

    y_g = torch.randn(32, 1, 16, 16)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    
    ls = []
    for epoch in range(10):
        model.train()
        # if epoch % 10 == 0:
        #     model.apply(reset_parameters)

        y_hat = model(graph, image, text)
        optimizer.zero_grad()
        y_hat = y_hat.permute((0, 3, 1, 2))

        loss = F.mse_loss(y_hat, y_g)
        loss.backward()
        optimizer.step()
        

        model.eval()
        with torch.no_grad():
            y_hat = model(graph, image, text)
            y_hat = y_hat.permute((0, 3, 2, 1))
            ls.append(F.mse_loss(y_hat, y_g).item())


    plt.plot(ls)
    plt.show()
    
