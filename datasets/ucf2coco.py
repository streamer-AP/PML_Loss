import numpy as np
import json
import os
import cv2
from tqdm import tqdm
from scipy import io as sio
from glob import glob

sub_dirs=["train","test"]
categories=[{
        "id":0,"name":"person","supercategory":"person"
    }
    ]
for sub_dir in sub_dirs:
    img_file_list=glob(f"{sub_dir}/img_*.jpg")
    mat_file_list=glob(f"{sub_dir}/img_*.mat")
    
    output_file=f"{sub_dir}_annotation.json"
    
    images=[]
    annotations=[]
    anno_id=0

    for img_path,mat_path in zip(img_file_list,mat_file_list):

        img=cv2.imread(img_path)
        mat=sio.loadmat(mat_path)
        file_name=os.path.basename(img_path)
        idx=img_path.split("/")[-1].split(".")[0].split("_")[-1]
        points=mat["annPoints"]
        
        obj={}
        obj["bbox"]=[]
        obj["iscrowd"]=[]
        
        for pt in points:
            obj["bbox"].append((pt[0],pt[1],1,1))
        
        images.append({
            "id":int(idx),
            "file_name":file_name,
            "width":img.shape[0],
            "height":img.shape[1],
        })
        for box in obj["bbox"]:
            annotations.append({
                "id":anno_id,
                "image_id":int(idx),
                "bbox":[box[0],box[1],box[2],box[3]],
                "iscrowd":0,
                "category_id":0,
                "area":(box[2])*(box[3])
            })
            anno_id+=1

    with open(output_file,"w") as f:
        json.dump({"annotations":annotations,"images":images,"categories":categories},f)