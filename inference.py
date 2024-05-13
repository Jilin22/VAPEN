import os
import os.path as osp
import time

import cv2
import numpy as np
import torchvision.transforms as T

from MyModel import *


model = VAPEN().cuda().eval()
weight_path = "ckpts/net_latest.pth"
model.load_state_dict(torch.load(weight_path)['params'])

test_path = "./imgs/low"
result_path = "./imgs/results_VAPEN"

if not osp.exists(result_path):
    os.makedirs(result_path)
    print(f"\t- [INFO] created out_dir: {result_path}")

img_list = os.listdir(test_path)

info = f"\t- [INFO] VAPEN processing, [{len(img_list)}] imgs in total."
print(info)

t = T.ToTensor()
dt = []

with torch.no_grad():
    for file in img_list:
        if not osp.isfile(osp.join(test_path, file)):
            continue
        img_path = osp.join(test_path, file)
        img_input = cv2.imread(img_path)
        img_hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
        h, s, v_input = cv2.split(img_hsv)
        tensor_input = t(v_input).cuda().unsqueeze(0)

        t_start = time.time()
        tensor_output, _ = model(tensor_input)
        dt.append(time.time() - t_start)

        v_out = tensor_output.squeeze().float().detach().cpu().clamp_(0, 1)
        v_out = (v_out.numpy() * 255.0).round().astype(np.uint8)

        img_output = cv2.merge([h, s, v_out])
        img_output = cv2.cvtColor(img_output, cv2.COLOR_HSV2BGR)
        cv2.imwrite(osp.join(result_path, file), img_output)

# discard the first forward propagation time
dt.pop(0)

print(f"\t- [INFO] average time: {np.mean(dt) * 1000:0.2f} ms.")
print("\t- [INFO] INFO: done!")
