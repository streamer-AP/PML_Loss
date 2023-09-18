from genericpath import exists
import os

import cv2
import einops
import numpy as np
import torch
from matplotlib import pyplot as plt
import random

def get_origin_img(img,std,mean):
    origin_input = einops.rearrange(img, "c h w->h w c")
    if origin_input.shape[2]>3:
        origin_input=origin_input[:,:,:3]
    origin_input = origin_input*torch.tensor([[std]]).to(
        origin_input.device)+torch.tensor([[mean]]).to(origin_input.device)
    origin_input = (origin_input.detach(
    ).cpu().numpy()*255).astype(np.uint8)
    origin_input = cv2.cvtColor(
        origin_input, cv2.COLOR_RGB2BGR)
    
    return origin_input
    

class Drawer_DenseMap():
    def __init__(self, args,is_ema=False) -> None:
        self.draw_freq = args.draw_freq
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir,exist_ok=True)
        self.draw_original = args.draw_original and not is_ema
        self.draw_denseMap = args.draw_denseMap and not is_ema
        self.draw_output = args.draw_output
        self.draw_point = args.draw_point
        self.mean = args.mean
        self.std = args.std
        self.cnt = 0
        self.is_ema="ema" if is_ema else "normal"

    def click(self):
        self.cnt += 1
        return self.cnt % self.draw_freq == 0
    def clear(self):
        self.cnt = 0

    def __call__(self,epoch=-1, inputs=None, targets=None, dMaps=None,oMaps=None,header="train"):
        if self.click():
            sub_dir = os.path.join(self.output_dir, f"epoch{epoch}")
            os.makedirs(sub_dir,exist_ok=True)
            j=random.randint(0,len(inputs)-1)
            if self.draw_original and inputs is not None:
                origin_input=get_origin_img(inputs[j],self.std,self.mean)
                cv2.imwrite(os.path.join(
                    sub_dir, f"{header}_{self.cnt}.jpg"), origin_input)
            draw_denseMap=self.draw_denseMap
            if targets is not None:
                if isinstance(targets,np.ndarray):
                    target_num=np.sum(targets[j])
                else:
                    target_num=targets[j]
                    draw_denseMap=False
            else:
                target_num=0
            if draw_denseMap and targets is not None:
                origin_input=get_origin_img(inputs[j],self.std,self.mean)
                origin_input=cv2.cvtColor(origin_input,cv2.COLOR_RGB2BGR)
                origin_input=origin_input.astype(np.float32)/255.0
                target_dMap = targets[j]/(0.0000001+np.max(targets[j]))
                target_dMap=target_dMap.transpose([1, 2, 0])
                
                stride=origin_input.shape[0]//target_dMap.shape[0]
                target_dMap=einops.repeat(target_dMap,"h w c->(h h2) (w w2) c",h2=stride,w2=stride)
                target_dMap=target_dMap*0.5+0.5*origin_input
                plt.figure(figsize=(10, 10))
                plt.imshow(target_dMap)
                plt.title(f"{header}_{target_num}")
                plt.savefig(os.path.join(
                    sub_dir, f"{header}_{self.cnt}_target.jpg"))
                plt.close()
            if self.draw_point and oMaps is not None:
                
                for k in range(len(oMaps[j])):
                    oMap=oMaps[j][k]
                    origin_input=get_origin_img(inputs[j],self.std,self.mean)
                    stride=origin_input.shape[-2]//oMap.shape[-1]

                    for l in range(oMap.shape[-2]):
                        for m in range(oMap.shape[-1]):
                            if oMap[0,l,m]>0:
                                y=torch.sigmoid(oMap[2,l,m])*stride+l*stride
                                x=torch.sigmoid(oMap[1,l,m])*stride+m*stride
                                cv2.circle(origin_input,(int(x),int(y)),4,(0,0,255),2)
                    origin_input=cv2.cvtColor(origin_input,cv2.COLOR_RGB2BGR)
                    origin_input=origin_input.astype(np.float32)/255.0
                    hot_map=torch.sigmoid(oMap[:1]).detach().cpu().numpy()
                    hot_map=einops.repeat(hot_map,"c h w->(h h2) (w w2) c",h2=stride,w2=stride,c=1)
                    hot_map=hot_map*0.5+0.5*origin_input
                    plt.figure(figsize=(10, 10))
                    plt.imshow(hot_map)
                    plt.title(f"{header}_{target_num}_oMap")
                    plt.savefig(os.path.join(
                        sub_dir, f"{header}_{self.cnt}_oMap_{k}.jpg"))
                    plt.close()
            if self.draw_output and dMaps is not None:
                origin_input=get_origin_img(inputs[j],self.std,self.mean)
                origin_input=cv2.cvtColor(origin_input,cv2.COLOR_RGB2BGR)
                origin_input=origin_input.astype(np.float32)/255.0
                for k in range(len(dMaps)):
                    output = dMaps[j][k]
                    if isinstance(output,torch.Tensor):
                        output=output.detach().cpu().numpy()
                    output_num=np.sum(output)
                    output=output/(0.0000001+np.max(output))
                    output=output.transpose([1,2,0])
                    stride=origin_input.shape[0]//output.shape[0]
                    output=einops.repeat(output,"h w c->(h h2) (w w2) c",h2=stride,w2=stride)
                    output=output*0.5+0.5*origin_input
                    plt.figure(figsize=(10, 10))
                    plt.imshow(output)
                    plt.title(f"{header}_{target_num}_{output_num}_dMap")
                    plt.savefig(os.path.join(
                        sub_dir, f"{header}_{self.cnt}_dMap_{k}.jpg"))
                    plt.close()
