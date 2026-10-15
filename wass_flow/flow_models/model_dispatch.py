import argparse
import os
import torch

def dispatch_model(net_model, vae, text_model, vision_model=None, num_device=None):
    ### 打印 模型参数
    if vae is not None:
        model_size = sum(p.data.nelement() for p in vae.parameters())
        print(f"Vae Model params: {model_size / 1024 / 1024:.2f} M")
    if text_model is not None:
        model_size = sum(p.data.nelement() for p in text_model.parameters())
        print(f"Text-CLIP Model params: {model_size / 1024 / 1024:.2f} M")
    if net_model is not None:
        model_size = sum(p.data.nelement() for p in net_model.parameters())
        print(f"Unet Model params: {model_size / 1024 / 1024:.2f} M")
        model_size = sum(p.data.nelement() for p in net_model.down_blocks.parameters())
        print(f"Unet Model-Down params: {model_size / 1024 / 1024:.2f} M")
        model_size = sum(p.data.nelement() for p in net_model.mid_block.parameters())
        print(f"Unet Model-Mid params: {model_size / 1024 / 1024:.2f} M")
        model_size = sum(p.data.nelement() for p in net_model.up_blocks.parameters())
        print(f"Unet Model-Up params: {model_size / 1024 / 1024:.2f} M")
    if vision_model is not None:
        model_size = sum(p.data.nelement() for p in vision_model.parameters())
        print(f"Vision Model params: {model_size / 1024 / 1024:.2f} M")

    ### 设备移动
    if num_device is None:
        num_device = torch.cuda.device_count()

    if num_device == 1:
        if text_model is not None:
            text_model = text_model.to('cuda:0')
        if vae is not None:
            vae = vae.to('cuda:0')
        if vision_model is not None:
            vision_model = vision_model.to('cuda:0')
            vision_model.device = torch.device('cuda:0')

        net_model = net_model.to('cuda:0')
        net_model.down_device = {0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:0', 3: 'cuda:0'}
        net_model.up_device = {0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:0', 3: 'cuda:0'}
        net_model.mid_device = 'cuda:0'
        
    elif num_device == 2:
        if text_model is not None:
            text_model = text_model.to('cuda:0')
        if vae is not None:
            vae = vae.to('cuda:0')
        if vision_model is not None:
            vision_model = vision_model.to('cuda:0')
            vision_model.device = torch.device('cuda:0')
        net_model = net_model.to('cuda:0')
        if net_model.down_blocks is not None:
            net_model.down_block = net_model.down_blocks.to('cuda:1')
            net_model.down_device = {0: 'cuda:1', 1: 'cuda:1', 2: 'cuda:1', 3: 'cuda:1'}
            print(net_model.down_device)
        if net_model.mid_block is not None:
            net_model.mid_block = net_model.mid_block.to('cuda:0')
            net_model.mid_device = 'cuda:0'
            print(net_model.mid_device)
        if net_model.up_blocks is not None:
            net_model.up_device = {0: 'cuda:1', 1: 'cuda:1', 2: 'cuda:1', 3: 'cuda:1'}
            net_model.up_blocks = net_model.up_blocks.to('cuda:1')
            print(net_model.up_device)

    elif num_device == 3:
        if text_model is not None:
            text_model = text_model.to('cuda:1')
        if vision_model is not None:
            vision_model = vision_model.to('cuda:1')
            vision_model.device = torch.device('cuda:1')
        if vae is not None:
            vae = vae.to('cuda:0')

        net_model = net_model.to('cuda:1')
        if net_model.down_blocks is not None:
            net_model.down_block = net_model.down_blocks.to('cuda:1')
            net_model.down_device = {0: 'cuda:1', 1: 'cuda:1', 2: 'cuda:1', 3: 'cuda:1'}
            print(net_model.down_device)
        if net_model.mid_block is not None:
            net_model.mid_block = net_model.mid_block.to('cuda:1')
            net_model.mid_device = 'cuda:1'
            print(net_model.mid_device)
        if net_model.up_blocks is not None:
            net_model.up_device = {0: 'cuda:1', 1: 'cuda:2', 2: 'cuda:2', 3: 'cuda:2'}
            net_model.up_blocks = net_model.up_blocks.to('cuda:1')
            print(net_model.up_device)
            
    elif num_device == 4:
        if text_model is not None:
            text_model = text_model.to('cuda:0')
        if vision_model is not None:
            vision_model = vision_model.to('cuda:0')
            vision_model.device = torch.device('cuda:0')
        if vae is not None:
            vae = vae.to('cuda:1')
        net_model = net_model.to('cuda:0')
        if net_model.down_blocks is not None:
            net_model.down_block = net_model.down_blocks.to('cuda:0')
            net_model.down_device = {0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:0', 3: 'cuda:0'}
            print(net_model.down_device)
        if net_model.mid_block is not None:
            net_model.mid_block = net_model.mid_block.to('cuda:0')
            net_model.mid_device = 'cuda:0'
            print(net_model.mid_device)
        if net_model.up_blocks is not None:
            net_model.up_device = {0: 'cuda:2', 1: 'cuda:2', 2: 'cuda:3', 3: 'cuda:3'}
            net_model.up_blocks = net_model.up_blocks.to('cuda:2')
            print(net_model.up_device)
            
    elif num_device == 5:
        if vision_model is not None:
            vision_model = vision_model.to('cuda:2')
            vision_model.device = torch.device('cuda:2')
        if text_model is not None:
            text_model = text_model.to('cuda:2')
        if vae is not None:
            vae = vae.to('cuda:0')
        net_model = net_model.to('cuda:2')
        if net_model.down_blocks is not None:
            net_model.down_device = {0: 'cuda:1', 1: 'cuda:1', 2: 'cuda:1', 3: 'cuda:1'}
            net_model.down_block = net_model.down_blocks.to('cuda:1')
            print(net_model.down_device)
        if net_model.mid_block is not None:
            net_model.mid_block = net_model.mid_block.to('cuda:2')
            net_model.mid_device = 'cuda:2'
            print(net_model.mid_device)
        if net_model.up_blocks is not None:
            net_model.up_device = {0: 'cuda:3', 1: 'cuda:3', 2: 'cuda:4', 3: 'cuda:4'}
            net_model.up_blocks = net_model.up_blocks.to('cuda:3')
            print(net_model.up_device)

    else:
        # 兜底分支: 当 GPU 数量 > 5 或其他未显式处理的情况时,
        # 默认按单卡 cuda:0 放置所有模型, 避免模型权重遗漏在 CPU 上
        # (曾导致 RuntimeError: Input type (torch.cuda.FloatTensor) and
        #  weight type (torch.FloatTensor) should be the same)
        print(f"[WARN] dispatch_model: num_device={num_device} 未显式处理, "
              f"回退到单卡 cuda:0 模式")
        if text_model is not None:
            text_model = text_model.to('cuda:0')
        if vae is not None:
            vae = vae.to('cuda:0')
        if vision_model is not None:
            vision_model = vision_model.to('cuda:0')
            vision_model.device = torch.device('cuda:0')

        net_model = net_model.to('cuda:0')
        net_model.down_device = {0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:0', 3: 'cuda:0'}
        net_model.up_device = {0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:0', 3: 'cuda:0'}
        net_model.mid_device = 'cuda:0'

    if vision_model is not None:
        return net_model, vae, text_model, vision_model

    return net_model, vae, text_model, None


