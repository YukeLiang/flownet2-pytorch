# insert current path to sys path
import os
import sys
sys.path.insert(0, os.getcwd())

import glob
import torch
import numpy as np
import argparse

from models import FlowNet2#the path is depended on where you create this module
from utils.frame_utils import read_gen#the path is depended on where you create this module 

def get_optical_flow(frame_list, output_dir):
    #obtain the necessary args for construct the flownet framework
    #initial a Net
    print("In function right now!")
    net = FlowNet2(argparse.Namespace(fp16=False, rgb_max=255.0)).cuda()
    #load the state_dict
    dict = torch.load("./FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    
    for idx,f in enumerate(frame_list):
        print("Generating flow for %d and %d" %(idx,idx+1))
        if idx == len(frame_list)-1: 
            break
        #load the image pair, you can find this operation in dataset.py
        img1 = read_gen(f)
        img2 = read_gen(frame_list[idx+1])
        images = [img1, img2]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        #process the image pair to obtian the flow 
        result = net(im).squeeze()
        data = result.data.cpu().numpy().transpose(1, 2, 0)

        # write file
        output_file = '%s%06d.flo'%(output_dir, idx+1)
        print("Writing to file: %s"%output_file)
        writeFlow(output_file,data) 
    
    return 'Finish'

#save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project 
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()

def main(video_dir, output_dir):
    final_status = False
    # list all subfolders, i.e., videos
    video_list = sorted(glob.glob(os.path.join(video_dir, '*/')))
    print('Found {:d} videos'.format(len(video_list)))
    
    #create output folder if necessary
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for video_folder in video_list:
        frame_list = sorted(glob.glob(os.path.join(video_folder, 'img', "*.jpg")))
    
        if len(frame_list) == 0:
        # empty folder
            print("{:s} empty folder!".format(video_folder))
            return True, 'Empty'
        else:
            print("{:d} frames in {:s}".format(len(frame_list), video_folder))
        
        print('READY TO GENEARTE OPTICAL FLOW!')
        get_optical_flow(frame_list, output_dir)

    final_status = True
    return final_status

##### For https://github.com/NVIDIA/flownet2-pytorch
if __name__ == '__main__':
    description = 'Helper script for running detector over video frames.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('video_dir', type=str,
                   help='video frames directory where each video has a folder.')
    p.add_argument('output_dir', type=str,
                   help='Output directory where detection results will be saved.') 
    args = vars(p.parse_args())
    main(args["video_dir"], args["output_dir"])