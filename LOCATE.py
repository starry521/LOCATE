import os
import argparse
from tqdm import tqdm
import cv2
import torch
import numpy as np
from models.locate import Net as model
from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map
from utils.evaluation import cal_kl, cal_sim, cal_nss
from data.datatest import TestData
import torch.utils.data
from yacs.config import CfgNode as CN
from Utils import *

class LOCATE:
    def __init__(self, cfg):
        self.cfg = cfg
        self._init_paths()
        self._init_aff_list()
        set_seed(seed=0)
        
        # init model
        self.model = model(aff_classes=self.cfg.num_classes).cuda()
        assert os.path.exists(self.cfg.model_file), "model file not found!"
        self.model.load_state_dict(torch.load(self.cfg.model_file))
        self.model.eval()
        
        # preocess GT data
        self.GT_path = f"{self.cfg.divide}_gt.t7"
        if not os.path.exists(self.GT_path):
            process_gt(self.cfg)
        self.GT_masks = torch.load(self.GT_path)
        
    def _init_paths(self):
        """init paths"""
        self.cfg.test_root = os.path.join(
            self.cfg.data_root, self.cfg.divide, "testset", "egocentric"
        )
        self.cfg.mask_root = os.path.join(
            self.cfg.data_root, self.cfg.divide, "testset", "GT"
        )
        if self.cfg.viz and not os.path.exists(self.cfg.save_path):
            os.makedirs(self.cfg.save_path, exist_ok=True)
            
    def _init_aff_list(self):
        """init affordance list"""
        if self.cfg.divide == "Seen":
            self.aff_list = [
                'beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", 
                "drag", 'drink_with', "eat", "hit", "hold", "jump", "kick", 
                "lie_on", "lift", "look_out", "open", "pack", "peel", "pick_up", 
                "pour", "push", "ride", "sip", "sit_on", "stick", "stir", 
                "swing", "take_photo", "talk_on", "text_on", "throw", "type_on", 
                "wash", "write"
            ]
            self.cfg.num_classes = 36
        else:
            self.aff_list = [
                "carry", "catch", "cut", "cut_with", 'drink_with', "eat", "hit", 
                "hold", "jump", "kick", "lie_on", "open", "peel", "pick_up", 
                "pour", "push", "ride", "sip", "sit_on", "stick", "swing", 
                "take_photo", "throw", "type_on", "wash"
            ]
            self.cfg.num_classes = 25
    
    def predict(self):
        """predict affordance"""
        testset = TestData(
            image_root=self.cfg.test_root,
            crop_size=self.cfg.crop_size,
            divide=self.cfg.divide, 
            mask_root=self.cfg.mask_root
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.test_num_workers,
            pin_memory=True
        )
        
        KLs, SIM, NSS = [], [], []
        for step, (image, label, mask_path) in enumerate(tqdm(test_loader)):
            # forward pass
            ego_pred = self.model.test_forward(image.cuda(), label.long().cuda())
            ego_pred = np.array(ego_pred.squeeze().data.cpu())
            ego_pred = normalize_map(ego_pred, self.cfg.crop_size)
            
            # load GT mask
            names = mask_path[0].split("/")
            key = f"{names[-3]}_{names[-2]}_{names[-1]}"
            GT_mask = self.GT_masks[key] / 255.0
            GT_mask = cv2.resize(GT_mask, (self.cfg.crop_size, self.cfg.crop_size))
            
            # cal metrics
            kld = cal_kl(ego_pred, GT_mask)
            sim = cal_sim(ego_pred, GT_mask)
            nss = cal_nss(ego_pred, GT_mask)
            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)
            
            # viz
            if self.cfg.viz:
                img_name = key.split(".")[0]
                viz_pred_test(
                    self.cfg, image, ego_pred, GT_mask, 
                    self.aff_list, label, img_name
                )
        
        # results
        results = {
            "KLD": round(sum(KLs)/len(KLs), 3),
            "SIM": round(sum(SIM)/len(SIM), 3),
            "NSS": round(sum(NSS)/len(NSS), 3)
        }
        return results


# if __name__ == '__main__':
#     # config
#     class Config:
#         data_root = './AGD20K/'
#         model_file = './checkpoints/best_seen.pth'
#         save_path = './save_preds'
#         divide = "Seen"
#         crop_size = 224
#         resize_size = 256
#         test_batch_size = 1
#         test_num_workers = 8
#         viz = True
    
#     cfg = Config()
#     locator = LOCATE(cfg)
#     metrics = locator.predict()
#     print(f"KLD = {metrics['KLD']}\nSIM = {metrics['SIM']}\nNSS = {metrics['NSS']}")

if __name__ == "__main__":

    # set work dir to LOCATE
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    input = Submodule()
    pkl_file = os.path.abspath("./data.pkl")
    input.deserialize(pkl_file)

    locate_cfg = input.get("locate_cfg", CN)

    locate = LOCATE(locate_cfg)
    metrics = locate.predict()

    input.clear()
    print(f"KLD = {metrics['KLD']}\nSIM = {metrics['SIM']}\nNSS = {metrics['NSS']}")