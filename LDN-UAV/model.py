


class FEM(nn.Module): # FEM
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1), nn.BatchNorm2d(in_channel), nn.Sigmoid())
        self.avg_pool = nn.AvgPool2d(kernel_size=3,padding=1,stride=1)
        self.avg5_pool = nn.AvgPool2d(kernel_size=5,padding=2,stride=1,)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel), nn.SiLU())

    def forward(self, x):
        adjust_weight = self.conv1(self.avg_pool(x) + self.avg5_pool(x))
        adjusted_feature = adjust_weight * x
        return self.conv2(adjusted_feature)


class AFEM(nn.Module):   # Adaptive FEM
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.avg5_pool = nn.AvgPool2d(kernel_size=5, padding=2, stride=1)

        self.adapt_pool = nn.AdaptiveAvgPool2d(1)
        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.adapt_conv = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=(kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0)),nn.Sigmoid())

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1), nn.BatchNorm2d(in_channel), nn.Sigmoid())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel), nn.SiLU())

    def forward(self, x):
        avg3 = self.avg_pool(x)
        avg5 = self.avg5_pool(x)

        adapt_pool3 = self.adapt_pool(avg3).unsqueeze(1)
        adapt_pool5 = self.adapt_pool(avg5).unsqueeze(1)
        adapt_weight = self.adapt_conv(torch.cat([adapt_pool3, adapt_pool5], 1))
        adapt_3, adapt_5 = torch.chunk(adapt_weight, 2, 1)
        data = avg3 * adapt_3.squeeze(1) + avg5 * adapt_5.squeeze(1)
        return self.conv2(self.conv1(data) * x)

class CSLConv(nn.Module):
    def __init__(self, in_channel, out_channel, real=7, stride=1):
        super().__init__()
        kernel_size = 3
        self.real = real
        self.kernel_size = kernel_size
        self.MLP = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * real, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * real),
            nn.ReLU())
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(real, 1), stride=(real, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c = x.shape[0:2]
        generat_feature = self.MLP(x)
        h, w = generat_feature.shape[2:]
        convdata = rearrange(generat_feature.view(b, c, self.real, h, w), 'b c size h w -> b c (h size) w')
        return self.act(self.bn(self.conv(convdata)))

class CAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.generate_feature = Conv(in_channel,out_channel*3,3,g=in_channel)
        self.get_weight = nn.Sequential(nn.Conv2d(out_channel*3,out_channel*3,1,groups=in_channel,bias=False))
        self.conv1 =  nn.Sequential(nn.Conv2d(out_channel,out_channel,1,groups=in_channel,bias=False),nn.Sigmoid())
        self.conv2 = Conv(out_channel*2,out_channel,1)
    def forward(self, x):
        b, c,h,w = x.shape
        feature = self.generate_feature(x)
        weight = self.get_weight(feature).view(b,c,3,h,w).softmax(2)
        data = torch.sum(feature.view(b,c,3,h,w) * weight,2)
        other = self.conv1(x) * x
        return self.conv2(torch.cat([data,other],1))


class IC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = FEM(c1, c_)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))