# from tkinter import X
# import torch
# import torch.nn as nn
# from thop import profile

# from timm.models.layers import DropPath
# import torch.nn.functional as F

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x, *args, **kwargs):
#         x = self.norm(x)
#         return self.fn(x, *args, **kwargs)


# class PReLU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight=nn.Parameter(torch.Tensor([.1]))
#     def forward(self, x):
#         return F.prelu(x,self.weight)

# class Tanh(nn.Module):
#     def forward(self, x):
#         return torch.tanh(x)
    
# def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size // 2), bias=bias, stride=stride)

# class FFN(nn.Module):
#     def __init__(self, in_dim,out_dim, mult=4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_dim, in_dim * mult, 1, 1, bias=False),
#             PReLU(),
#             nn.Conv2d(in_dim * mult, in_dim * mult, 3, 1, 1,
#                       bias=False, groups=in_dim * mult),
#             PReLU(),
#             nn.Conv2d(in_dim * mult, out_dim, 1, 1, bias=False),
#         )

#     def forward(self, x):
#         """
#         x: [b,c,h,w]
#         return out: [b,c,h,w]
#         """
#         return  self.net(x)
# ##########################################################################
# #####Global Convolutional Module
# class GCM(nn.Module):
#     def __init__(self,dim,kernel_size=4,stride=4,
#                  drop_path=0.,ffn_ratio=4, norm_layer=nn.LayerNorm,bias=False):
#         super(GCM, self).__init__()
#         #out_dim=out_dim*stride
#         self.down=nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,stride=1, padding=0, bias=bias),
#                                 nn.UpsamplingBilinear2d(scale_factor=1.0/stride))
#         self.norm1=norm_layer(dim)
#         self.block=nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,stride=1,padding=0,groups=1,bias=bias),
#                                 nn.Conv2d(dim,dim,kernel_size=kernel_size,stride=stride,padding=(kernel_size//2),groups=dim,bias=bias))
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # 不懂droppath是什么
#         self.reweight= FFN(dim, dim*2,ffn_ratio)

#         self.norm2=norm_layer(dim)
#         self.ffn = FFN(dim,dim,ffn_ratio)
        
#     def forward(self, x):
#         x=self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)
#         # 对通道进行归一化
#         d_x=self.down(x)
#         # 使用下采样和双线性上采样处理输入
#         x=self.drop_path(self.block(x))
#         f=x+d_x
#         # 使用加法组合缩减采样和处理的特征
#         B,C,H,W=f.shape
#         f=F.adaptive_avg_pool2d(f,output_size=1)
#         # 每个通道被压缩为一个标量，表示该通道的全局平均值。
#         f=self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
#         x=f[0]*x+f[1]*d_x
#         # 为什么f0给x f1给下采样的dx且这个矩阵是如何相乘且变换维度的
#         x=x+self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2))
#         del d_x,f
#         return x
    
# class ISP_Estimator(nn.Module):
#     def __init__(
#             self, dim,scale=2,norm_layer=nn.LayerNorm):  #__init__部分是内部属性，而forward的输入才是外部输入
#         super(ISP_Estimator, self).__init__()
#         self.norm=norm_layer(dim)
#         hidden_dim=dim*scale
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 3,1,1,bias=False),
#             PReLU(),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=7//2, bias=True, groups=dim),
#             PReLU(),
#             nn.Conv2d(hidden_dim, dim, 3,1,1, bias=True))
        
#     def forward(self, x):
#         x=x+self.net(self.norm(x.permute(0,2,3,1)).permute(0,3,1,2))
#         return x
    
# class CrossPatchModule(nn.Module):
#     def __init__(self, patches=[8,8]):
#         super().__init__()
#         self.patch_h=patches[0]
#         self.patch_w=patches[1]
#         self.patch_n=self.patch_h*self.patch_w
#         self.step=self.patch_n//self.patch_n
#         #pos embed
#         absolute_pos_embed = nn.Parameter(torch.zeros(1,self.step, self.patch_n, self.patch_n,1,1))
#         self.abs_pos=nn.init.trunc_normal_(absolute_pos_embed, std=.02)
#     #     使用截断的正态分布初始化
#     def forward(self,x):
#         B,C,H,W=x.shape

#         kernel_h=H//self.patch_h
#         kernel_w=W//self.patch_w
#         ###slicing and recombination
#         #print('cross')
#         unfold=torch.nn.Unfold(kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
#         x=unfold(x) #[N, C*kernel_size*kernel_size, self.patch_h*self.patch_w]
#         x=x.view(B,-1,kernel_h,kernel_w,self.patch_h,self.patch_w) #[N, C, kernel_size, kernel_size, patch_h, patch_w]
#         x=x.view(B,-1,kernel_h,kernel_w,self.patch_h*self.patch_w).permute(0,1,4,2,3)#[N, C, patch_h*patch_w(patch_n), kernel_size, kernel_size]
#         x=x.view(B, self.step, x.shape[1]//self.step, self.patch_n, kernel_h, kernel_w)
#         x=x+self.abs_pos
#         #rolling_cross_channel_cross_patch
#         for st in range(self.step): #step means groups
#             for m in range(self.patch_n):
#                 idx_i=[] # changing index list
#                 for i in range(m,self.patch_n):
#                     idx_i.append(i) #eg. (6,64)
#                 if m>0:
#                     for j in range(m): 
#                         idx_i.append(j) #eg (0,6) idx_i=[6,7,8,...,64,0,1,2,3,4,5]
#                 x[:,st,m,:]=x[:,st,m,idx_i] #change patches order
#         x=x.view(B,-1,self.patch_n, kernel_h, kernel_w) # [B,C,patch_num,kernel_h,kernel_w]
#         x=x.permute(0,1,3,4,2).view(B,-1,kernel_h*kernel_w,self.patch_h*self.patch_w)
#         x=x.view(B,-1,self.patch_h*self.patch_w)
#         fold=torch.nn.Fold(output_size=(H,W),kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
#         x=fold(x)
#         return x
    
# class LEB_sub(nn.Module):
#     def __init__(self, in_dim,out_dim,bias=False,proj_drop=0.,ffn_ratio=4):
#         super().__init__()
 
#         self.gate=nn.Sequential(nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias),
#                                 Tanh(),
#                                 nn.Conv2d(out_dim,out_dim,kernel_size=9,stride=1,padding=9//2,groups=out_dim,bias=bias)
#                                 )
#         self.wise_features=nn.Sequential(nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias),
#                                 PReLU(),
#                                 nn.Conv2d(out_dim,out_dim,kernel_size=9,stride=1,padding=9//2,groups=out_dim,bias=bias)
#                                 )

#         self.reweight = FFN(out_dim, out_dim * 2, ffn_ratio)
#         self.proj = nn.Conv2d(out_dim, out_dim, 1, 1,bias=True)
#         self.proj_drop = nn.Dropout(proj_drop)   
 
#     def forward(self, x):
     
#         x_wf=self.wise_features(x)
#         x_gate=self.gate(x)
#         f=x_wf+x_gate

#         B, C, H, W = f.shape
#         f=F.adaptive_avg_pool2d(f,output_size=1)
#         f=self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
#         x=f[0]*x_gate+f[1]*x_wf

#         del x_wf,x_gate,f
#         x = self.proj(x)
#         x = self.proj_drop(x)           
#         return x

# class LEB(nn.Module):

#     def __init__(self, dim, ffn_ratio=2., bias=False, 
#                  drop_path=0., norm_layer=nn.LayerNorm,):
#         super().__init__()
#         self.norm1=norm_layer(dim)
#         self.block = LEB_sub(dim,dim,bias=bias,ffn_ratio=ffn_ratio)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.ffn = FFN(dim, dim, ffn_ratio)
        
#     def forward(self,x):
#         x=x+self.drop_path(self.block(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)))
#         x =x+self.drop_path(self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)))
#         return x

# class UKNet(nn.Module):
#     """ UKNet Network with selective infrared small target enhancement """
#     def __init__(self, g_layers=3, l_layers=3, patches=[8,8], in_dim=3,
#                  feat_dim=96, ffn_ratio=2, bias=False, drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm):

#         super().__init__()
#         #####  wave_UNet responsible for constructing wave-like features
#         global_branch=[]
        
#         self.conv_feat=nn.Conv2d(in_dim,feat_dim,3,1,1,bias=False)
#         self.conv_g=nn.Conv2d(feat_dim,patches[0]*patches[1],3,1,1,bias=False)
#         self.cpm=CrossPatchModule(patches=patches)
#         self.global_cfc=nn.Conv2d(patches[0]*patches[1],feat_dim,kernel_size=1,stride=1,padding=0)
        
#         for block in range(5):
#             block = GCM(feat_dim,kernel_size=7,stride=2,ffn_ratio=ffn_ratio)
#             global_branch.append(block)
#         self.global_branch = nn.ModuleList(global_branch)

#         self.ch_ex1=nn.Conv2d(feat_dim//16,256,3,1,1,bias=False)
#         self.ch_ex2=nn.Conv2d(4,feat_dim,1,1,0)
        
#         net_g=[]
#         net_l=[]
        
#         for block in range(g_layers):
#             block = ISP_Estimator(dim=feat_dim,norm_layer=norm_layer)
#             net_g.append(block)
#         self.net_g=nn.ModuleList(net_g)
       
#         for block in range(l_layers):
#             block = LEB(dim=feat_dim,ffn_ratio=ffn_ratio,norm_layer=norm_layer)
#             net_l.append(block)
#         self.net_l=nn.ModuleList(net_l)
          
#         self.out_g=nn.Conv2d(feat_dim,3,3,1,1,bias=True)
#         self.out_l=nn.Conv2d(feat_dim,3,3,1,1,bias=True)
        
#     def forward(self, x, sky_mask, point_label):
#         """
#         x: 原始红外图像
#         sky_mask: 形状 [B,1,H,W]，天空区域为 1，其他区域为 0
#         point_label: 形状 [B,1,H,W]，红外小目标位置为 1，其他区域为 0
#         """
#         # 1. **只增强天空区域**
#         x = x * sky_mask  # 仅保留天空区域，其余部分置为 0

#         # 2. **标准的 UKNet 低光增强过程**
#         l_x=self.conv_feat(x)
#         g_x=self.conv_g(l_x)
        
#         ######Net_l
#         for idx, block in enumerate(self.net_l):
#             l_x = block(l_x)
        
#         ######Net_g
#         B, C, H, W = g_x.shape
#         g_x = self.cpm(g_x)
#         g_x = self.global_cfc(g_x)

#         for idx, block in enumerate(self.global_branch):
#             g_x = block(g_x)
            
#         B, c, h, w = g_x.shape
#         g_x = g_x.reshape(B, c//16, h*4, w*4)
#         g_x = self.ch_ex1(g_x)
        
#         B, c, h, w = g_x.shape
#         scale = (H//h) * (W//w)
#         g_x = g_x.reshape(B, c//scale, h*(H//h), w*(W//w))
        
#         g_x = self.ch_ex2(g_x) #[B,C,H,W]
        
#         #### project global info
#         for idx,block in enumerate(self.net_g):
#             g_x = block(g_x)

#         # 3. **只增强红外小目标**
#         x = self.out_g(g_x) * x + self.out_l(l_x)
#         x = x * point_label + x.detach() * (1 - point_label)  # 只增强目标区域，背景保持原样
        
#         return x
    
# if __name__=='__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model=UKNet()
#     model = model.to(device)
#     if str(device) =='cuda':
#         input=torch.randn(2,3,256,256).cuda()
#     else:
#         input=torch.randn(2,3,256,256)
#     print(model)
#     flops,params=profile(model,inputs=(input,))
#     print('flops:{}G params:{}M'.format(flops/1e9,params/1e6))
# # from tkinter import X
# # import torch
# # import torch.nn as nn
# # from thop import profile

# # from timm.models.layers import DropPath
# # import torch.nn.functional as F

# # class PreNorm(nn.Module):
# #     def __init__(self, dim, fn):
# #         super().__init__()
# #         self.fn = fn
# #         self.norm = nn.LayerNorm(dim)

# #     def forward(self, x, *args, **kwargs):
# #         x = self.norm(x)
# #         return self.fn(x, *args, **kwargs)


# # class PReLU(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.weight=nn.Parameter(torch.Tensor([.1]))
# #     def forward(self, x):
# #         return F.prelu(x,self.weight)

# # class Tanh(nn.Module):
# #     def forward(self, x):
# #         return torch.tanh(x)
    
# # def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
# #     return nn.Conv2d(
# #         in_channels, out_channels, kernel_size,
# #         padding=(kernel_size // 2), bias=bias, stride=stride)

# # class FFN(nn.Module):
# #     def __init__(self, in_dim,out_dim, mult=4):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Conv2d(in_dim, in_dim * mult, 1, 1, bias=False),
# #             PReLU(),
# #             nn.Conv2d(in_dim * mult, in_dim * mult, 3, 1, 1,
# #                       bias=False, groups=in_dim * mult),
# #             PReLU(),
# #             nn.Conv2d(in_dim * mult, out_dim, 1, 1, bias=False),
# #         )

# #     def forward(self, x):
# #         """
# #         x: [b,c,h,w]
# #         return out: [b,c,h,w]
# #         """
# #         return  self.net(x)
# # ##########################################################################
# # #####Global Convolutional Module
# # class GCM(nn.Module):
# #     def __init__(self,dim,kernel_size=4,stride=4,
# #                  drop_path=0.,ffn_ratio=4, norm_layer=nn.LayerNorm,bias=False):
# #         super(GCM, self).__init__()
# #         #out_dim=out_dim*stride
# #         self.down=nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,stride=1, padding=0, bias=bias),
# #                                 nn.UpsamplingBilinear2d(scale_factor=1.0/stride))
# #         self.norm1=norm_layer(dim)
# #         self.block=nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,stride=1,padding=0,groups=1,bias=bias),
# #                                 nn.Conv2d(dim,dim,kernel_size=kernel_size,stride=stride,padding=(kernel_size//2),groups=dim,bias=bias))
# #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# #         # 不懂droppath是什么
# #         self.reweight= FFN(dim, dim*2,ffn_ratio)

# #         self.norm2=norm_layer(dim)
# #         self.ffn = FFN(dim,dim,ffn_ratio)
        
# #     def forward(self, x):
# #         x=self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)
# #         # 对通道进行归一化
# #         d_x=self.down(x)
# #         # 使用下采样和双线性上采样处理输入
# #         x=self.drop_path(self.block(x))
# #         f=x+d_x
# #         # 使用加法组合缩减采样和处理的特征
# #         B,C,H,W=f.shape
# #         f=F.adaptive_avg_pool2d(f,output_size=1)
# #         # 每个通道被压缩为一个标量，表示该通道的全局平均值。
# #         f=self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
# #         x=f[0]*x+f[1]*d_x
# #         # 为什么f0给x f1给下采样的dx且这个矩阵是如何相乘且变换维度的
# #         x=x+self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2))
# #         del d_x,f
# #         return x
    
# # class ISP_Estimator(nn.Module):
# #     def __init__(
# #             self, dim,scale=2,norm_layer=nn.LayerNorm):  #__init__部分是内部属性，而forward的输入才是外部输入
# #         super(ISP_Estimator, self).__init__()
# #         self.norm=norm_layer(dim)
# #         hidden_dim=dim*scale
# #         self.net = nn.Sequential(
# #             nn.Conv2d(dim, hidden_dim, 3,1,1,bias=False),
# #             PReLU(),
# #             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=7//2, bias=True, groups=dim),
# #             PReLU(),
# #             nn.Conv2d(hidden_dim, dim, 3,1,1, bias=True))
        
# #     def forward(self, x):
# #         x=x+self.net(self.norm(x.permute(0,2,3,1)).permute(0,3,1,2))
# #         return x
    
# # class CrossPatchModule(nn.Module):
# #     def __init__(self, patches=[8,8]):
# #         super().__init__()
# #         self.patch_h=patches[0]
# #         self.patch_w=patches[1]
# #         self.patch_n=self.patch_h*self.patch_w
# #         self.step=self.patch_n//self.patch_n
# #         #pos embed
# #         absolute_pos_embed = nn.Parameter(torch.zeros(1,self.step, self.patch_n, self.patch_n,1,1))
# #         self.abs_pos=nn.init.trunc_normal_(absolute_pos_embed, std=.02)
# #     #     使用截断的正态分布初始化
# #     def forward(self,x):
# #         B,C,H,W=x.shape

# #         kernel_h=H//self.patch_h
# #         kernel_w=W//self.patch_w
# #         ###slicing and recombination
# #         #print('cross')
# #         unfold=torch.nn.Unfold(kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
# #         x=unfold(x) #[N, C*kernel_size*kernel_size, self.patch_h*self.patch_w]
# #         x=x.view(B,-1,kernel_h,kernel_w,self.patch_h,self.patch_w) #[N, C, kernel_size, kernel_size, patch_h, patch_w]
# #         x=x.view(B,-1,kernel_h,kernel_w,self.patch_h*self.patch_w).permute(0,1,4,2,3)#[N, C, patch_h*patch_w(patch_n), kernel_size, kernel_size]
# #         x=x.view(B, self.step, x.shape[1]//self.step, self.patch_n, kernel_h, kernel_w)
# #         x=x+self.abs_pos
# #         #rolling_cross_channel_cross_patch
# #         for st in range(self.step): #step means groups
# #             for m in range(self.patch_n):
# #                 idx_i=[] # changing index list
# #                 for i in range(m,self.patch_n):
# #                     idx_i.append(i) #eg. (6,64)
# #                 if m>0:
# #                     for j in range(m): 
# #                         idx_i.append(j) #eg (0,6) idx_i=[6,7,8,...,64,0,1,2,3,4,5]
# #                 x[:,st,m,:]=x[:,st,m,idx_i] #change patches order
# #         x=x.view(B,-1,self.patch_n, kernel_h, kernel_w) # [B,C,patch_num,kernel_h,kernel_w]
# #         x=x.permute(0,1,3,4,2).view(B,-1,kernel_h*kernel_w,self.patch_h*self.patch_w)
# #         x=x.view(B,-1,self.patch_h*self.patch_w)
# #         fold=torch.nn.Fold(output_size=(H,W),kernel_size=(kernel_h,kernel_w),stride=(kernel_h,kernel_w))
# #         x=fold(x)
# #         return x
    
# # class LEB_sub(nn.Module):
# #     def __init__(self, in_dim,out_dim,bias=False,proj_drop=0.,ffn_ratio=4):
# #         super().__init__()
 
# #         self.gate=nn.Sequential(nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias),
# #                                 Tanh(),
# #                                 nn.Conv2d(out_dim,out_dim,kernel_size=9,stride=1,padding=9//2,groups=out_dim,bias=bias)
# #                                 )
# #         self.wise_features=nn.Sequential(nn.Conv2d(in_dim,out_dim,1,1,0,bias=bias),
# #                                 PReLU(),
# #                                 nn.Conv2d(out_dim,out_dim,kernel_size=9,stride=1,padding=9//2,groups=out_dim,bias=bias)
# #                                 )

# #         self.reweight = FFN(out_dim, out_dim * 2, ffn_ratio)
# #         self.proj = nn.Conv2d(out_dim, out_dim, 1, 1,bias=True)
# #         self.proj_drop = nn.Dropout(proj_drop)   
 
# #     def forward(self, x):
     
# #         x_wf=self.wise_features(x)
# #         x_gate=self.gate(x)
# #         f=x_wf+x_gate

# #         B, C, H, W = f.shape
# #         f=F.adaptive_avg_pool2d(f,output_size=1)
# #         f=self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
# #         x=f[0]*x_gate+f[1]*x_wf

# #         del x_wf,x_gate,f
# #         x = self.proj(x)
# #         x = self.proj_drop(x)           
# #         return x

# # class LEB(nn.Module):

# #     def __init__(self, dim, ffn_ratio=2., bias=False, 
# #                  drop_path=0., norm_layer=nn.LayerNorm,):
# #         super().__init__()
# #         self.norm1=norm_layer(dim)
# #         self.block = LEB_sub(dim,dim,bias=bias,ffn_ratio=ffn_ratio)
# #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# #         self.norm2 = norm_layer(dim)
# #         self.ffn = FFN(dim, dim, ffn_ratio)
        
# #     def forward(self,x):
# #         x=x+self.drop_path(self.block(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)))
# #         x =x+self.drop_path(self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)))
# #         return x

# # class UKNet(nn.Module):
# #     """ UKNet Network """
# #     def __init__(self, g_layers=3,l_layers=3,patches=[8,8],in_dim=3,
# #         feat_dim=96,ffn_ratio=2,
# #         bias=False, drop_path_rate=0.,
# #         norm_layer=nn.LayerNorm):

# #         super().__init__()
# #         #####  wave_UNet responsible for constructing wave-like features
# #         global_branch=[]
        
# #         self.conv_feat=nn.Conv2d(in_dim,feat_dim,3,1,1,bias=False)
# #         self.conv_g=nn.Conv2d(feat_dim,patches[0]*patches[1],3,1,1,bias=False)
# #         self.cpm=CrossPatchModule(patches=patches)
# #         self.global_cfc=nn.Conv2d(patches[0]*patches[1],feat_dim,kernel_size=1,stride=1,padding=0)
        
# #         for block in range(5):
# #             block = GCM(feat_dim,kernel_size=7,stride=2,ffn_ratio=ffn_ratio)
            
# #             global_branch.append(block)
# #         self.global_branch = nn.ModuleList(global_branch)

# #         self.ch_ex1=nn.Conv2d(feat_dim//16,256,3,1,1,bias=False)
# #         self.ch_ex2=nn.Conv2d(4,feat_dim,1,1,0)
        
# #         net_g=[]
# #         net_l=[]
        
# #         for block in range(g_layers):
# #             block = ISP_Estimator(dim=feat_dim,norm_layer=norm_layer)
            
# #             net_g.append(block)
# #         self.net_g=nn.ModuleList(net_g)
       
# #         for block in range(l_layers):
# #             block = LEB(dim=feat_dim,ffn_ratio=ffn_ratio,norm_layer=norm_layer)
            
# #             net_l.append(block)
# #         self.net_l=nn.ModuleList(net_l)
          
# #         self.out_g=nn.Conv2d(feat_dim,3,3,1,1,bias=True)
# #         self.out_l=nn.Conv2d(feat_dim,3,3,1,1,bias=True)
        
# #     def forward(self, x):
# #         # x_V = x.max(1,keepdim=True)[0]
# #         # in_faet=torch.cat([x,x_V],dim=1)
# #         # del x_V
# #         l_x=self.conv_feat(x)
# #         g_x=self.conv_g(l_x)
# #         ######Net_l
# #         for idx,block in enumerate(self.net_l):
# #             l_x=block(l_x)
# #         ######Net_g
# #                 ###### extract global info
# #         B,C,H,W=g_x.shape
        
# #         g_x=self.cpm(g_x)
# #         g_x= self.global_cfc(g_x)

# #         for idx,block in enumerate(self.global_branch):
# #             g_x=block(g_x)
            
# #         B,c,h,w=g_x.shape
# #         # g_x.shape [B,32,h,w]->[B,2,h*4,w*4]
# #         g_x=g_x.reshape(B,c//16,h*4,w*4)
# #         g_x=self.ch_ex1(g_x)
        
# #         B,c,h,w=g_x.shape
# #         scale=(H//h)*(W//w)
# #         g_x=g_x.reshape(B,c//scale,h*(H//h),w*(W//w))
        
# #         #[B,256,h*4,w*4] ->[B,c//scale,H,W]
# #         g_x=self.ch_ex2(g_x) #[B,C,H,W]
# #         #### project global info
# #         for idx,block in enumerate(self.net_g):
# #             g_x=block(g_x)

# #         x=self.out_g(g_x)*x+self.out_l(l_x)
# #         del g_x,l_x
        
# #         return x
    
# # if __name__=='__main__':
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     model=UKNet()
# #     model = model.to(device)
# #     if str(device) =='cuda':
# #         input=torch.randn(2,3,256,256).cuda()
# #     else:
# #         input=torch.randn(2,3,256,256)
# #     print(model)
# #     flops,params=profile(model,inputs=(input,))
# #     print('flops:{}G params:{}M'.format(flops/1e9,params/1e6))
from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# --- 兼容 timm 的 DropPath 新旧导入 ---
try:
    from timm.layers import DropPath
except Exception:
    from timm.models.layers import DropPath


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([.1]))

    def forward(self, x):
        return F.prelu(x, self.weight)


class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * mult, 1, 1, bias=False),
            PReLU(),
            nn.Conv2d(in_dim * mult, in_dim * mult, 3, 1, 1, bias=False, groups=in_dim * mult),
            PReLU(),
            nn.Conv2d(in_dim * mult, out_dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


##########################################################################
# Global Convolutional Module
class GCM(nn.Module):
    def __init__(self, dim, kernel_size=7, stride=2,
                 drop_path=0., ffn_ratio=4, norm_layer=nn.LayerNorm, bias=False):
        super(GCM, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.UpsamplingBilinear2d(scale_factor=1.0 / stride)
        )
        self.norm1 = norm_layer(dim)
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2), groups=dim, bias=bias)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.reweight = FFN(dim, dim * 2, ffn_ratio)
        self.norm2 = norm_layer(dim)
        self.ffn = FFN(dim, dim, ffn_ratio)

    def forward(self, x):
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        d_x = self.down(x)
        x = self.drop_path(self.block(x))
        f = x + d_x

        B, C, H, W = f.shape
        f = F.adaptive_avg_pool2d(f, output_size=1)
        f = self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = f[0] * x + f[1] * d_x
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        del d_x, f
        return x


class ISP_Estimator(nn.Module):
    def __init__(self, dim, scale=2, norm_layer=nn.LayerNorm):
        super(ISP_Estimator, self).__init__()
        self.norm = norm_layer(dim)
        hidden_dim = dim * scale
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False),
            PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=7 // 2, bias=True, groups=dim),
            PReLU(),
            nn.Conv2d(hidden_dim, dim, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        x = x + self.net(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class CrossPatchModule(nn.Module):
    def __init__(self, patches=[8, 8]):
        super().__init__()
        self.patch_h = patches[0]
        self.patch_w = patches[1]
        self.patch_n = self.patch_h * self.patch_w
        self.step = self.patch_n // self.patch_n
        absolute_pos_embed = nn.Parameter(torch.zeros(1, self.step, self.patch_n, self.patch_n, 1, 1))
        self.abs_pos = nn.init.trunc_normal_(absolute_pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        kernel_h = H // self.patch_h
        kernel_w = W // self.patch_w

        unfold = torch.nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        x = unfold(x)  # [N, C*kh*kw, patch_n]
        x = x.view(B, -1, kernel_h, kernel_w, self.patch_h, self.patch_w)
        x = x.view(B, -1, kernel_h, kernel_w, self.patch_n).permute(0, 1, 4, 2, 3)
        x = x.view(B, self.step, x.shape[1] // self.step, self.patch_n, kernel_h, kernel_w)
        x = x + self.abs_pos

        for st in range(self.step):
            for m in range(self.patch_n):
                idx_i = list(range(m, self.patch_n)) + list(range(0, m))
                x[:, st, m, :] = x[:, st, m, idx_i]

        x = x.view(B, -1, self.patch_n, kernel_h, kernel_w)
        x = x.permute(0, 1, 3, 4, 2).view(B, -1, kernel_h * kernel_w, self.patch_h * self.patch_w)
        x = x.view(B, -1, self.patch_h * self.patch_w)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        x = fold(x)
        return x


class LEB_sub(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, proj_drop=0., ffn_ratio=4):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias),
            Tanh(),
            nn.Conv2d(out_dim, out_dim, kernel_size=9, stride=1, padding=9 // 2, groups=out_dim, bias=bias)
        )
        self.wise_features = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias),
            PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=9, stride=1, padding=9 // 2, groups=out_dim, bias=bias)
        )

        self.reweight = FFN(out_dim, out_dim * 2, ffn_ratio)
        self.proj = nn.Conv2d(out_dim, out_dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_wf = self.wise_features(x)
        x_gate = self.gate(x)
        f = x_wf + x_gate

        B, C, H, W = f.shape
        f = F.adaptive_avg_pool2d(f, output_size=1)
        f = self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = f[0] * x_gate + f[1] * x_wf

        del x_wf, x_gate, f
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LEB(nn.Module):
    def __init__(self, dim, ffn_ratio=2., bias=False, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.block = LEB_sub(dim, dim, bias=bias, ffn_ratio=ffn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ffn = FFN(dim, dim, ffn_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.block(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        x = x + self.drop_path(self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class UKNet(nn.Module):
    """ UKNet Network """
    def __init__(self, g_layers=3, l_layers=3, patches=[8, 8], in_dim=3,
                 feat_dim=96, ffn_ratio=2, bias=False, drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # 公共第一层：x -> F
        self.conv_feat = nn.Conv2d(in_dim, feat_dim, 3, 1, 1, bias=False)

        # 右支起点：把 F 投到 patch_n 通道，供 CPM 使用
        self.conv_g = nn.Conv2d(feat_dim, patches[0] * patches[1], 3, 1, 1, bias=False)
        self.cpm = CrossPatchModule(patches=patches)
        self.global_cfc = nn.Conv2d(patches[0] * patches[1], feat_dim, kernel_size=1, stride=1, padding=0)

        # 全局分支（GCM×5）
        global_branch = []
        for _ in range(5):
            block = GCM(feat_dim, kernel_size=7, stride=2, ffn_ratio=ffn_ratio)
            global_branch.append(block)
        self.global_branch = nn.ModuleList(global_branch)

        # 通道/空间整形
        self.ch_ex1 = nn.Conv2d(feat_dim // 16, 256, 3, 1, 1, bias=False)
        self.ch_ex2 = nn.Conv2d(4, feat_dim, 1, 1, 0)

        # 右支后续（把全局语义“STM”起来）
        net_g = []
        for _ in range(g_layers):
            block = ISP_Estimator(dim=feat_dim, norm_layer=norm_layer)
            net_g.append(block)
        self.net_g = nn.ModuleList(net_g)

        # 左支（LEM 堆叠）
        net_l = []
        for _ in range(l_layers):
            block = LEB(dim=feat_dim, ffn_ratio=ffn_ratio, norm_layer=norm_layer)
            net_l.append(block)
        self.net_l = nn.ModuleList(net_l)

        # 两个头：左支输出、右支输出
        self.out_g = nn.Conv2d(feat_dim, 3, 3, 1, 1, bias=True)
        self.out_l = nn.Conv2d(feat_dim, 3, 3, 1, 1, bias=True)

    # 原始前向（保留）
    def forward(self, x):
        l_x = self.conv_feat(x)
        g_x = self.conv_g(l_x)

        # 左支
        for block in self.net_l:
            l_x = block(l_x)

        # 右支
        B, C, H, W = g_x.shape
        g_x = self.cpm(g_x)
        g_x = self.global_cfc(g_x)
        for block in self.global_branch:
            g_x = block(g_x)

        B, c, h, w = g_x.shape
        g_x = g_x.reshape(B, c // 16, h * 4, w * 4)
        g_x = self.ch_ex1(g_x)

        B, c, h, w = g_x.shape
        scale = (H // h) * (W // w)
        g_x = g_x.reshape(B, c // scale, h * (H // h), w * (W // w))
        g_x = self.ch_ex2(g_x)

        for block in self.net_g:
            g_x = block(g_x)

        x = self.out_g(g_x) * x + self.out_l(l_x)
        del g_x, l_x
        return x

    # —— 新增：严格按图(a)的前向 —— #
    @torch.no_grad()
    def forward_diagram(self, x, return_all=True):
        """
        x -> conv_feat -> F
        左支:  F -> net_l -> out_l -> L
        右支:  F -> conv_g -> CPM -> global_cfc -> GCM×5 -> reshape/extract -> ISP×g -> out_g -> G
        融合:  Y = L * sigmoid(G)
        """
        # 1) 公共特征 F
        Fm = self.conv_feat(x)  # [B, feat_dim, H, W]

        # 2) 左支：LEM 堆叠 -> out_l
        l_x = Fm
        for block in self.net_l:
            l_x = block(l_x)
        L = self.out_l(l_x)  # 左支要参与乘法的 3 通道

        # 3) 右支：PSM(CPM) -> GCMx5 -> ISPxg -> out_g
        g0 = self.conv_g(Fm)         # 先记下 H0, W0
        B0, C0, H0, W0 = g0.shape

        G_psm = self.cpm(g0)         # 仅作中间保存
        g_x = self.global_cfc(G_psm)

        for block in self.global_branch:
            g_x = block(g_x)

        B, c, h, w = g_x.shape
        g_x = g_x.reshape(B, c // 16, h * 4, w * 4)
        g_x = self.ch_ex1(g_x)

        B, c, h, w = g_x.shape
        scale = (H0 // h) * (W0 // w)
        g_x = g_x.reshape(B, c // scale, h * (H0 // h), w * (W0 // w))
        g_x = self.ch_ex2(g_x)

        G_stm = g_x
        for block in self.net_g:
            G_stm = block(G_stm)
        G = self.out_g(G_stm)        # 右支 3 通道“增益/门控”

        # 4) 融合（逐元素相乘；增益用 sigmoid 约束到 (0,1)）
        Y = L * torch.sigmoid(G)

        if return_all:
            return {
                "F": Fm,
                "L_lem": l_x,
                "L": L,
                "G_psm": G_psm,
                "G_stm": G_stm,
                "G": G,
                "Y": Y
            }
        else:
            return Y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UKNet().to(device)
    if str(device) == 'cuda':
        inp = torch.randn(2, 3, 256, 256).cuda()
    else:
        inp = torch.randn(2, 3, 256, 256)
    print(model)
    flops, params = profile(model, inputs=(inp,))
    print('flops:{}G params:{}M'.format(flops / 1e9, params / 1e6))
