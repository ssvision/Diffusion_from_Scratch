#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Explaining and better commented/annotated for understanding code of 
Original U-Net Model as per Explaining AI youtube video
'''


import torch
import torch.nn as nn

class TemporalEncoding(nn.Module):

    ''' This method of embedding time is not in the OG DDPM Paper
        This is taken from the Stable Diffusion Paper
    '''
    def __init__(self, time_step, time_embed_dim):
        super().__init__()

        ''' inputs are :
            time_step = 1-D Tensor of size B=batch_size
            time_embed_dim = (int) dimension of the embeddings
        '''
        self.time_step = time_step
        self.time_embed_dim = time_embed_dim
        self.temporal_embeddings = None
        assert self.time_embed_dim % 2 == 0, "Time Embedding Dimension must be even"

    @staticmethod
    def get_temporal_embeddings(time_step, time_embed_dim):
        assert time_embed_dim % 2 == 0, "Time Embedding Dimension must be even"

        indices = torch.arange(time_embed_dim // 2, dtype=torch.float32)
        normalised_indices = indices / (time_embed_dim // 2)
        factors = 10000 ** normalised_indices

        t_emb = time_step[:, None].repeat(1, time_embed_dim // 2) / factors
        temporal_embeddings = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

        ''' Returns temporal embeddings of size BxD'''
        return temporal_embeddings

    def _explanation(self):
        ''' if let's assume timestep=3; B=5, D=8
        time_step = tensor([3, 3, 3, 3, 3])
        factors = tensor([1., 10., 100., 1000.])
        t_emb = tensor([[3.0000, 0.3000, 0.0300, 0.0030],
                        [3.0000, 0.3000, 0.0300, 0.0030],
                        [3.0000, 0.3000, 0.0300, 0.0030],
                        [3.0000, 0.3000, 0.0300, 0.0030],
                        [3.0000, 0.3000, 0.0300, 0.0030]])
        temporal_embeddings = tensor([[ 0.1411,  0.2955,  0.0300,  0.0030, -0.9900,  0.9553,  0.9996,  1.0000],
                                      [ 0.1411,  0.2955,  0.0300,  0.0030, -0.9900,  0.9553,  0.9996,  1.0000],
                                      [ 0.1411,  0.2955,  0.0300,  0.0030, -0.9900,  0.9553,  0.9996,  1.0000],
                                      [ 0.1411,  0.2955,  0.0300,  0.0030, -0.9900,  0.9553,  0.9996,  1.0000],
                                      [ 0.1411,  0.2955,  0.0300,  0.0030, -0.9900,  0.9553,  0.9996,  1.0000]])
        '''
        pass


class DownBlock(nn.Module):
    '''
    This is the Downblock module contains 'n' layers of Downblock layers
    followed by downsampling of images
    Each Downblock layer consists of 
    1. ResNet Layer (Norm, Silu, Conv)
    2. Time embedding layer (Silu, FFN)
    3. Attention (Norm, Self Attention)
    There are residual connections as well
    Refer to reference material for details
    '''

    def __init__(self, in_channels, out_channels, time_embed_dim, 
                 down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers


        '''
        This is the input to the 1st layer and then subsequent layers of the downblock module 
        '''
        self.downblock_input = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.temb_proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )

        '''Conv is used for downsampling however pooling can also be used'''
        #self.down_sample = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2) if self.down_sample else nn.Identity()

    def forward(self, X, time_embeddings):

        out=X
        for i in range(self.num_layers):

            # 1st part ResNet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.temb_proj_layers[i](time_embeddings)[:,:,None,None]
            out = self.resnet_conv_second[i](out) 
            out = out + self.downblock_input[i](resnet_input)

            # 2nd part Attention
            batch_size, channels, h, w = out.shape
            in_attention = out.reshape(batch_size, channels, h*w)
            in_attention = self.attention_norms[i](in_attention)
            in_attention = in_attention.transpose(1,2)
            attn_ops, _ = self.attentions[i](in_attention, in_attention, in_attention)
            attn_ops = attn_ops.transpose(1,2).view(batch_size, channels, h, w)
            attn_ops = attn_ops + out

            
        # Final Downsampling
        out = self.down_sample(out)
        return out


class MidBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim,num_heads=4,num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.outer_resnet_conv_1 = nn.Sequential(
                            nn.GroupNorm(8, in_channels),
                            nn.SiLU(),
                            nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=1, padding=1)
                            )
        self.outer_resnet_conv_2 = nn.Sequential(
                            nn.GroupNorm(8, out_channels),
                            nn.SiLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                            )
        
        self.outer_residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.inner_residual_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.temb_proj_layers = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(time_embed_dim, out_channels)
                            )
        self.inner_resnet_conv_1 = nn.Sequential(
                                nn.GroupNorm(8,out_channels),
                                nn.SiLU(),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                                )
        self.inner_resnet_conv_2 = nn.Sequential(
                                nn.GroupNorm(8,out_channels),
                                nn.SiLU(),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                                )
        self.inner_attention_norm = nn.GroupNorm(8, out_channels)
        self.inner_attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        
        self.time_embedding_Layers = nn.ModuleList([self.temb_proj_layers for _ in range(self.num_layers)])
        self.attention_norm_Layers = nn.ModuleList([self.inner_attention_norm for _ in range(self.num_layers)])
        self.attention_Layers = nn.ModuleList([self.inner_attention for _ in range(self.num_layers)])
        self.inner_resnet_1_Layers = nn.ModuleList([self.inner_resnet_conv_1 for _ in range(self.num_layers)])
        self.inner_resnet_2_Layers = nn.ModuleList([self.inner_resnet_conv_2 for _ in range(self.num_layers)])
        self.inner_residual_Layers = nn.ModuleList([self.inner_residual_layer for _ in range(self.num_layers)])

    def forward(self, X, time_embeddings):

        outer_block_input = X
        out_ = self.outer_resnet_conv_1(X)
        out_ = out_ + self.temb_proj_layers(time_embeddings)[:,:,None,None]
        out_ = self.outer_resnet_conv_2(out_)
        out_ = out_ + self.outer_residual_layer(outer_block_input)

        for i in range(self.num_layers):
            inner_block_input = out_ if i==0 else out

            #attention block
            b, c, h, w = out_.shape
            in_attn = out_.reshape(b,c,h*w)
            in_attn = self.attention_norm_Layers[i](in_attn)
            in_attn = in_attn.transpose(1,2)
            in_attn,_ = self.attention_Layers[i](in_attn, in_attn, in_attn)
            in_attn = in_attn.transpose(1,2).view(b,c,h,w)
            out = in_attn + inner_block_input

            #resnet block
            resnet_ip = out
            out = self.inner_resnet_1_Layers[i](out)
            out = out + self.time_embedding_Layers[i](time_embeddings)[:,:,None,None]
            out = self.inner_resnet_2_Layers[i](out)
            out = out + self.inner_residual_Layers[i](resnet_ip)
        
        return out


class UpBlock(nn.Module):
    '''
    This is the Downblock module contains 'n' layers of Downblock layers
    followed by downsampling of images
    Each Downblock layer consists of 
    1. ResNet Layer (Norm, Silu, Conv)
    2. Time embedding layer (Silu, FFN)
    3. Attention (Norm, Self Attention)
    There are residual connections as well
    Refer to reference material for details
    '''

    def __init__(self, in_channels, out_channels, time_embed_dim, 
                 up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.up_sample = up_sample
        self.num_layers = num_layers


        '''
        This is the input to the 1st layer and then subsequent layers of the upblock module 
        '''
        self.upblock_input = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.temb_proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )

        '''Conv is used for downsampling however pooling can also be used'''
        self.up_sample = nn.ConvTranspose2d(in_channels//2, in_channels//2, 4, 2, 1) if self.up_sample else nn.Identity()
        

    def forward(self, X, out_downblock, time_embeddings):

        X = self.up_sample(X)
        X = torch.cat([X, out_downblock], dim=1)
        out = X

        for i in range(self.num_layers):

            # 1st part ResNet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.temb_proj_layers[i](time_embeddings)[:,:,None,None]
            out = self.resnet_conv_second[i](out) 
            out = out + self.upblock_input[i](resnet_input)

            # 2nd part Attention
            batch_size, channels, h, w = out.shape
            in_attention = out.reshape(batch_size, channels, h*w)
            in_attention = self.attention_norms[i](in_attention)
            in_attention = in_attention.transpose(1,2)
            attn_ops, _ = self.attentions[i](in_attention, in_attention, in_attention)
            attn_ops = attn_ops.transpose(1,2).view(batch_size, channels, h, w)
            attn_ops = attn_ops + out

        
        return out


class Unet(nn.Module):
    # """
    # Unet model comprising
    # Down blocks, Midblocks and Uplocks
    # """
        def __init__(self, model_config=None):
            super().__init__()


            # im_channels = model_config['im_channels']
            # self.down_channels = model_config['down_channels']
            # self.mid_channels = model_config['mid_channels']
            # self.t_emb_dim = model_config['time_emb_dim']
            # self.down_sample = model_config['down_sample']
            # self.num_down_layers = model_config['num_down_layers']
            # self.num_mid_layers = model_config['num_mid_layers']
            # self.num_up_layers = model_config['num_up_layers']

            im_channels = 3
            self.down_channels = [32,64,128,256]
            self.mid_channels = [256,256,128]
            self.t_emb_dim = 128
            self.down_sample = [True, True, False]
            self.num_down_layers = 2
            self.num_mid_layers = 2
            self.num_up_layers = 2

           


            
            assert self.mid_channels[0] == self.down_channels[-1]
            assert self.mid_channels[-1] == self.down_channels[-2]
            assert len(self.down_sample) == len(self.down_channels) - 1
            
            # Initial projection from sinusoidal time embedding
            self.t_proj = nn.Sequential(
                nn.Linear(self.t_emb_dim, self.t_emb_dim),
                nn.SiLU(),
                nn.Linear(self.t_emb_dim, self.t_emb_dim)
            )

            self.up_sample = list(reversed(self.down_sample))
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
            
            self.downs = nn.ModuleList([])
            for i in range(len(self.down_channels)-1):
                self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                            down_sample=self.down_sample[i], num_layers=self.num_down_layers))
            
            self.mids = nn.ModuleList([])
            for i in range(len(self.mid_channels)-1):
                self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                        num_layers=self.num_mid_layers))
            
            self.ups = nn.ModuleList([])
            for i in reversed(range(len(self.down_channels)-1)):
                self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                        self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers))
            
            self.norm_out = nn.GroupNorm(8, 16)
            self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
        
        def forward(self, x, t):
            # Shapes assuming downblocks are [C1, C2, C3, C4]
            # Shapes assuming midblocks are [C4, C4, C3]
            # Shapes assuming downsamples are [True, True, False]
            # B x C x H x W
            out = self.conv_in(x)
            # B x C1 x H x W
            
            # t_emb -> B x t_emb_dim
            t_emb = TemporalEncoding.get_temporal_embeddings(torch.as_tensor(t).long(), self.t_emb_dim)

            # t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
            t_emb = self.t_proj(t_emb)
            
            down_outs = []
            
            for idx, down in enumerate(self.downs):
                down_outs.append(out)
                out = down(out, t_emb)
            # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
            # out B x C4 x H/4 x W/4
                
            for mid in self.mids:
                out = mid(out, t_emb)
            # out B x C3 x H/4 x W/4
            
            for up in self.ups:
                down_out = down_outs.pop()
                # print(f"down_out is : {down_out.shape}")
                out = up(out, down_out, t_emb)
                
                # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
                
            out = self.norm_out(out)
            out = nn.SiLU()(out)
            out = self.conv_out(out)
            # out B x C x H x W
            return out






if __name__ == "__main__":


    class Unet(nn.Module):
    # """
    # Unet model comprising
    # Down blocks, Midblocks and Uplocks
    # """
        def __init__(self, model_config=None):
            super().__init__()


            # im_channels = model_config['im_channels']
            # self.down_channels = model_config['down_channels']
            # self.mid_channels = model_config['mid_channels']
            # self.t_emb_dim = model_config['time_emb_dim']
            # self.down_sample = model_config['down_sample']
            # self.num_down_layers = model_config['num_down_layers']
            # self.num_mid_layers = model_config['num_mid_layers']
            # self.num_up_layers = model_config['num_up_layers']

            im_channels = 1
            self.down_channels = [32,64,128,256]
            self.mid_channels = [256,256,128]
            self.t_emb_dim = 128
            self.down_sample = [True, True, False]
            self.num_down_layers = 2
            self.num_mid_layers = 2
            self.num_up_layers = 2

           


            
            assert self.mid_channels[0] == self.down_channels[-1]
            assert self.mid_channels[-1] == self.down_channels[-2]
            assert len(self.down_sample) == len(self.down_channels) - 1
            
            # Initial projection from sinusoidal time embedding
            self.t_proj = nn.Sequential(
                nn.Linear(self.t_emb_dim, self.t_emb_dim),
                nn.SiLU(),
                nn.Linear(self.t_emb_dim, self.t_emb_dim)
            )

            self.up_sample = list(reversed(self.down_sample))
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
            
            self.downs = nn.ModuleList([])
            for i in range(len(self.down_channels)-1):
                self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                            down_sample=self.down_sample[i], num_layers=self.num_down_layers))
            
            self.mids = nn.ModuleList([])
            for i in range(len(self.mid_channels)-1):
                self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                        num_layers=self.num_mid_layers))
            
            self.ups = nn.ModuleList([])
            for i in reversed(range(len(self.down_channels)-1)):
                self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                        self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers))
            
            self.norm_out = nn.GroupNorm(8, 16)
            self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
        
        def forward(self, x, t):
            # Shapes assuming downblocks are [C1, C2, C3, C4]
            # Shapes assuming midblocks are [C4, C4, C3]
            # Shapes assuming downsamples are [True, True, False]
            # B x C x H x W
            out = self.conv_in(x)
            # B x C1 x H x W
            
            # t_emb -> B x t_emb_dim
            t_emb = TemporalEncoding.get_temporal_embeddings(torch.as_tensor(t).long(), self.t_emb_dim)

            # t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
            t_emb = self.t_proj(t_emb)
            
            down_outs = []
            
            for idx, down in enumerate(self.downs):
                down_outs.append(out)
                out = down(out, t_emb)
            # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
            # out B x C4 x H/4 x W/4
                
            for mid in self.mids:
                out = mid(out, t_emb)
            # out B x C3 x H/4 x W/4
            
            for up in self.ups:
                down_out = down_outs.pop()
                # print(f"down_out is : {down_out.shape}")
                out = up(out, down_out, t_emb)
                
                # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
                
            out = self.norm_out(out)
            out = nn.SiLU()(out)
            out = self.conv_out(out)
            # out B x C x H x W
            return out
        
    
    torch.manual_seed(123)
    model = Unet()
    input = torch.randn(2,1,28,28)
    time = torch.full((2,),5)
    out = model(input, time)
    print(f"Shape of op is : {out.shape}")





















































    '''####################################################################################'''
    # Experimenting here
    # torch.manual_seed(123)
    # batch_size = 2
    # down_block = DownBlock(in_channels=8, out_channels=16, time_embed_dim=10)
    # tenc = TemporalEncoding(time_step = torch.full((batch_size,),5), time_embed_dim=10)
    # time_embeddings = tenc.get_temporal_embeddings()
    # input = torch.randn(batch_size, 8, 64, 32) # ip of 2 batches each of 8x64x32 each
    # print(f"Shape of Input is : {input.shape}")
    # output_downblock = down_block(input, time_embeddings)
    # print(f"Shape of output of DOWNBLOCK  : {output_downblock.shape}")
    # mid_block = MidBlock(in_channels=16, out_channels=32, time_embed_dim=10)
    # output_midblock = mid_block(output_downblock, time_embeddings)
    # print(f"Shape of output of MIDBLOCK  : {output_midblock.shape}")

    # random_ = torch.randn(batch_size,32,64,32)
    # up_block = UpBlock(in_channels=32*2, out_channels=16, time_embed_dim=10)
    # output_upblock = up_block(output_midblock, random_, time_embeddings)
    # print(f"Shape of output of UPBLOCK  : {output_upblock.shape}")






    '''####################################################################################'''
    # Experimenting here
    # tenc = TemporalEncoding(time_step = torch.full((2,),5), time_embed_dim= 10)
    # time_embeddings = tenc.get_temporal_embeddings()
    # print(f"Embeddings are : {time_embeddings}")
    # print(f"Shape of Embeddings are : {embeddings.shape}")

    # temb_proj_layers = nn.Sequential(
    #             nn.SiLU(),
    #             nn.Linear(10, 4)
    #         )
    # out_1 = temb_proj_layers(embeddings)
    # print(f"shape is : {out_1.shape}")

    # out_2 = out_1[:,:,None,None]
    # print(f"shape is : {out_2.shape}")

    '''#######################################################################################'''
    # '''Experimenting with one layer'''
    # torch.manual_seed(123)


    # in_channels=8
    # out_channels=16
    # time_embed_dim=10
    # num_heads=4
    # num_batch = 2

    # tenc = TemporalEncoding(time_step = torch.full((2,),5), time_embed_dim= 10)
    # time_embeddings = tenc.get_temporal_embeddings()
    # print(f"Embeddings are : {time_embeddings.shape}")
    

    
    # X = torch.randn(num_batch,in_channels,128,64) # assume img size to be 128x64 

    # '''
    # This is the input to the 1st layer and then subsequent layers of the downblock module 
    # '''
    # downblock_input = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    # resnet_conv_first = nn.Sequential(
    #             nn.GroupNorm(8, in_channels),
    #             nn.SiLU(),
    #             nn.Conv2d(in_channels, out_channels,
    #                         kernel_size=3, stride=1, padding=1),
    #         )
            
    # op_resnet_conv_first = resnet_conv_first(X)
    # print(f"O/P of 1st ResNet layer is : {op_resnet_conv_first.shape}")
    

    # temb_proj_layers = nn.Sequential(
    #         nn.SiLU(),
    #         nn.Linear(time_embed_dim, out_channels)
    #     )
    
    # op_temb_proj_layers = temb_proj_layers(time_embeddings)
    # print(f"O/P of Time Embeddings Projection Layer is : {op_temb_proj_layers.shape}")

    # ip_resnet_conv_second = op_resnet_conv_first + op_temb_proj_layers[:,:,None,None]
    
  

    # resnet_conv_second = nn.Sequential(
    #             nn.GroupNorm(8, out_channels),
    #             nn.SiLU(),
    #             nn.Conv2d(out_channels, out_channels,
    #                         kernel_size=3, stride=1, padding=1),
    #         )
    
    # op_resnet_conv_second = resnet_conv_second(ip_resnet_conv_second)
    # print(f"O/P of 2st ResNet layer is : {op_resnet_conv_second.shape}")

    
    # op_ResNet_Block = op_resnet_conv_second + downblock_input(X)
    # print(f"O/P of 2st ResNet BLOCK is : {op_ResNet_Block.shape}")

    # attention_norms = nn.GroupNorm(8, out_channels)
            
    # attentions = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

    # batch_size, channels, h, w = op_ResNet_Block.shape
    # in_attn_block = op_ResNet_Block.reshape(batch_size, channels, h*w)

    # op_attn_norm_block = attention_norms(in_attn_block)
    # print(f"O/P of attention norm block is : {op_attn_norm_block.shape}")

    # ip_attn_block = op_attn_norm_block.transpose(1,2)
    # op_attn_block, _ = attentions(ip_attn_block, ip_attn_block, ip_attn_block)
    # print(f"O/P of attention block is : {op_attn_block.shape}")

    # op_attn_block = op_attn_block.transpose(1,2).view(batch_size, channels, h, w)
    # op_attn_block = op_attn_block + op_ResNet_Block
    # print(f"Final O/P of attention block is : {op_attn_block.shape}")

    # # '''Conv is used for downsampling however pooling can also be used'''
    # # #down_sample = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()
    # down_sample = nn.AvgPool2d(kernel_size=2, stride=2) 
    # print(f"Final O/P after DownSampling is : {down_sample(op_attn_block).shape}")
    '''###################################################################################'''
            
    