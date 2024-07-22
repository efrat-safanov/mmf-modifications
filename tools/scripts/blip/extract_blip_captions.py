from PIL import Image
import argparse
import os
from glob import glob
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import csv



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_IMAGE_SIZE = 384
MODEL_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

model = blip_decoder(pretrained=MODEL_URL, image_size=TARGET_IMAGE_SIZE, vit='base')
model.eval()
model = model.to(DEVICE)

def load_image(img_file, image_size=TARGET_IMAGE_SIZE):

    raw_image = Image.open(img_file).convert("RGB")

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(DEVICE)
    return image


def get_caption(image):
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        result = ""
        for c in caption:
            print('caption: ' + c)
            result += c + "."
        return result


def extract_dataset_captions(image_dir, name_to_id_map_file, results_file):
    image_list = glob(image_dir + "/*.*")
    image_list = {f: 1 for f in image_list}
    exclude = {}
    name_to_id_map = {}
    use_feat_name_as_id = False
    with open(name_to_id_map_file) as f:
        lines = f.readlines()
        for line in lines:
            line_arr = line.strip("\n").split(",")
            name_to_id_map[line_arr[1]] = line_arr[0].strip("\s")
            if line_arr[1] == "original_image_name":
                use_feat_name_as_id = True
                print("using feat name as id")
    fields = ['id', 'image_text']

    # writing to csv file
    with open(results_file, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

        image_list = list(image_list.keys())
        for n_im, impath in enumerate(image_list):
            if (n_im + 1) % 100 == 0:
                print("processing %d / %d" % (n_im + 1, len(image_list)))
            image_name = os.path.basename(impath)
            try:
                image_id = name_to_id_map[image_name]
            except Exception:
                print("id error for: " + image_name)
                continue

            try:
                img_data = load_image(impath)
                r = get_caption(img_data)
                csvwriter.writerow([image_id, r])
            except Exception:
                print("BLIP error for" + image_name)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--name_to_id_map", type=str, required=True, default="name_to_id_map.csv")

    args = parser.parse_args()

    extract_dataset_captions(args.image_dir, args.name_to_id_map, args.results_file)
