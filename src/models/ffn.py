import torch
from torch import nn


class FeedForward(nn.Module):
    """
    ## FFN module
    前馈神经网络，两层全连接网络，d_model -> d_ff -> d_model
    # d_ff 一般是 d_model 的四倍
    # 通过is_gated参数可以选择是否使用门控机制，即GLU（Gated Linear Units）
    # activation参数可以选择不同的激活函数，默认是ReLU
    # bias1, bias2, bias_gate参数可以选择是否在对应的线性层中使用偏置项
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding，d_model是词嵌入的特征数量
        * `d_ff` is the number of features in the hidden layer of the FFN，d_ff是FFN隐藏层的特征数量
        * `dropout` is dropout probability for the hidden layer,隐藏层的dropout概率
        * `is_gated` specifies whether the hidden layer is gated,是否使用门控机制
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1) # 线形层,输入层->隐藏层,d_model->d_ff,bias1表示是否使用偏置
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2) # 线形层,隐藏层->输出层,d_ff->d_model,bias2表示是否使用偏置
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)  # 随机失活层,防止过拟合
        # Activation function $f$
        self.activation = activation  # 激活函数
        # Whether there is a gate
        self.is_gated = is_gated  # 是否使用门控机制
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)  # 线形层,输入层->门控层,d_model->d_ff,bias_gate表示是否使用偏置
            # Gated-FFN(x)=Wo*​(σ(Wg*​x)⊙(Wv*​x))+b ,其中σ是激活函数，⊙是逐元素乘法，bias_gate表示是否使用偏置
    
    # define the forward pass,定义前向传播过程
    def forward(self, x: torch.Tensor):
        # f(x W_1 + b_1), 对输入x进行线性变换后再经过激活函数,f表示激活函数,layer1表示线性层
        g = self.activation(self.layer1(x))
        # If gated, f(x W_1 + b_1) ⊙ (x V + b) = g ⊙ (x V + b)
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout,  应用随机失活
        x = self.dropout(x)
        
        # 有门控机制时，输出为(f(x W_1 + b_1) ⊙ (x V + b)) W_2 + b_2 
        # 没有门控机制时，输出为f(x W_1 + b_1) W_2 + b_2
        # depending on whether it is gated
        return self.layer2(x)