from PIL import Image
import argparse
import os
from glob import glob
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import csv
import re

#for every file with the name split-xxx-id.png - get captions in a csv merged with original csv

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
            result += c + "."
        result = re.sub("((with|and) )?(a|the)? (capt\s|words|quote|text).*", "", c)
        words = result.split()
        if len(words) > 2 and words[-1] == words[-2]:
           result = re.sub(words[-1],  "", result).strip()
           result += " " + words[-1]
        print('caption: ' + result)
        return result


def extract_dataset_captions(image_dir, name_to_id_map_file, results_file, current_captions):

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
    curr_captions_data = {}

    with open(current_captions) as f:
        curr_captions_file = csv.reader(f)
        for row in curr_captions_file:
            curr_captions_data[row[0]] = row[1]

    # writing to csv file
    with open(results_file, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for image_name in name_to_id_map:
            id = name_to_id_map[image_name]
            if id == "id":
                continue

            file_id = str(int(id))# + 7000)
            specific_image_files = image_dir + "/" + str(file_id) + "-*.*"
            print("running id: " + str(id) + " with " + specific_image_files)
            image_list = glob(specific_image_files)
            image_list = {f: 1 for f in image_list}
            image_list = list(image_list.keys())

            split_captions = {}
            for n_im, impath in enumerate(image_list):
                curr_image_name = os.path.basename(impath)
                try:
                    img_data = load_image(impath)
                    r = get_caption(img_data)
                    split_captions[impath.split("-")[-1].split(".")[0]] = r
                except Exception:
                    print("BLIP error for" + curr_image_name)
                    continue
            if len(split_captions) > 1 and split_captions["000"] != split_captions["001"]:
                label = split_captions["000"] + " vs " + split_captions["001"]
                if len(split_captions) > 2:
                    label += split_captions["000"] + ", " + split_captions["001"] + ", " + split_captions["002"]
            else:
                label = curr_captions_data[id]
            csvwriter.writerow([id, label])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--name_to_id_map", type=str, required=True, default="name_to_id_map.csv")
    parser.add_argument("--current_caption", type=str, required=True, default="labels.csv")

    args = parser.parse_args()

    extract_dataset_captions(args.image_dir, args.name_to_id_map, args.results_file, args.current_caption)
