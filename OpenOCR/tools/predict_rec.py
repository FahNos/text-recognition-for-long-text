import os
import sys
import argparse
import math
import yaml 

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.engine.trainer import Trainer

class Recognizer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = Config(args.config)
        
        if args.opt:
            self.cfg.merge_dict(args.opt)

        self.cfg.cfg['Eval'] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cfg['Global']['device'] == 'gpu' else "cpu")
        
        trainer = Trainer(self.cfg, mode='eval', task='rec')
        
        self.model = trainer.model
        self.model.to(self.device)
        self.model.eval()

        self.post_process_class = trainer.post_process_class
        self.logger = trainer.logger
        self.logger.info("Recognizer model loaded successfully.")

        try:
            train_config = self.cfg.cfg['Train']
            sampler_config = train_config['sampler']
            dataset_config = train_config['dataset']
            
            self.base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]

            self.base_h = 32

            self.max_ratio = train_config['loader'].get('max_ratio', 25)

            self.padding = dataset_config.get('padding', False) 
            
        except KeyError as e:
            self.logger.error(f"Missing critical preprocessing parameter in config: {e}")
            raise

    def preprocess(self, image_path):      
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Cannot open image file: {image_path}. Error: {e}")
            return None

        w, h = img.size        
        aspect_ratio = w / h        
      
        gen_ratio = int(np.around(aspect_ratio))
        gen_ratio = max(1, min(gen_ratio, self.max_ratio))   

        imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
            self.base_h * gen_ratio, self.base_h
        ]      
  

        resized_image = img.resize((imgW, imgH), Image.BICUBIC)

        transform_to_tensor = T.ToTensor()
        img_tensor = transform_to_tensor(resized_image)      
        
        transform_normalize = T.Normalize(0.5, 0.5)
        img_tensor = transform_normalize(img_tensor)
        
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)


    def __call__(self, image_path):
        preprocessed_image = self.preprocess(image_path)
        if preprocessed_image is None:
            return "Error processing image", 0.0

        with torch.no_grad():
            preds = self.model(preprocessed_image, data=None)

        results = self.post_process_class(preds, batch=None)
        
        text, confidence = "", 0.0
        if isinstance(results, list) and len(results) > 0:
            main_results = results[0] 
            if isinstance(main_results, list) and len(main_results) > 0:
                text, confidence = main_results[0]
        
        return text, confidence

def _parse_opt(opts):
    config = {}
    if not opts:
        return config
    for s in opts:
        s = s.strip()
        k, v = s.split('=', 1)
        keys = k.split('.')
        d = config
        for i, key in enumerate(keys[:-1]):
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = yaml.load(v, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("-o", "--opt", nargs='*', help="Set configuration options. e.g., Global.pretrained_model=./model.pth")
    
    args = parser.parse_args()
    args.opt = _parse_opt(args.opt)
    
    recognizer = Recognizer(args)
    
    recognized_text, confidence_score = recognizer(args.image_path)
    
    print("="*50)
    print(f"Image Path: {args.image_path}")
    print(f"Recognized Text: {recognized_text}")
    print(f"Confidence: {confidence_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()