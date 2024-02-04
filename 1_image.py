from transformers import (
    AutoModel,
    CLIPProcessor
)
import cv2
import pandas as pd
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

def process_image_ed(pil_img: Image) -> Image:
    '''Traditional methods'''
    # Convert RGB to BGR
    open_cv_image = np.array(pil_img.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Increase contrast
    img_yuv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    contrast_increased = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Remove noise and smooth
    noise_removed = cv2.GaussianBlur(contrast_increased, (3, 3), 0)
    smooth_img = cv2.bilateralFilter(noise_removed, 9, 75, 75)
    erode_kernel = np.ones((5, 5), np.uint8)
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(smooth_img, dilate_kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, erode_kernel, iterations=1)

    # Resize (for speed)
    final_img = Image.fromarray(cv2.cvtColor(eroded_img, cv2.COLOR_BGR2RGB))
    final_img = final_img.resize((100, 100))

    return final_img


if __name__ == "__main__":

    # ARM Mac agnostic
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    src_dir = 'test/images'
    query_dir = 'queries/queries'
    train_dir = 'train/train'
    submission = pd.read_csv('sample_submission.csv')
    # Choosing a random big model
    model = AutoModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').to(device).eval()
    processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    # Unmatched = 22
    submission['dot_class'] = 22

    with torch.no_grad():
        query_images = []
        query_classes = []
        # Also make it predict the classes in the train dataset
        for file in tqdm(list(Path(query_dir).rglob('*.jpg'))+ list(Path(train_dir).rglob('*.jpg'))):
            inputs = processor(images=[process_image_ed(Image.open(file).convert('RGB'))], return_tensors='pt').to(device)
            outputs = model.get_image_features(inputs.pixel_values).cpu()
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            query_images.append(outputs)
            if 'queries' in str(file):
                query_classes.append(file.stem) # 0 - 21
            elif 'train' in str(file):
                # Assuming set of train labels does not intersect query
                query_classes.append(22)
        
        query_images = torch.cat(query_images) # 22 query + 2378 train = 2400

        for idx, row in tqdm(list(submission.iterrows())):
            if not pd.isna(row['class']):
                continue
            inputs = processor(images=[process_image_ed(Image.open(Path(src_dir) / row['img_file']).convert('RGB'))], return_tensors='pt').to(device)
            outputs = model.get_image_features(inputs.pixel_values).cpu()
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            values = outputs @ query_images.T

            # If all are equally dissimilar, then softmax would be ~ 1/n.
            # Add a confidence hyperparameter 1.2
            if values.softmax(1).max() > (1.2 * 1/2400):
                amax = values.argmax().item()
                submission.at[idx, 'dot_class'] = query_classes[amax]

        sub = submission[['img_file',]]
        sub['class'] = submission['dot_class']

    sub.to_csv('submission_0400.csv', index=False)

