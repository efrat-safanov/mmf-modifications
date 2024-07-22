from PIL import Image
import os
from glob import glob
import csv
import argparse
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_caption(args, text_generator, img_path):

    image_features = text_generator.get_img_feature([img_path], None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
    best_caption = "A" + captions[best_clip_idx]
    print("image: " + img_path)
    print(captions)
    print('best clip:', best_caption)
    return best_caption


def extract_dataset_captions(args, image_dir, name_to_id_map_file, results_file):
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

    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))
    existing_ids = {}
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_arr = line.strip("\n").split(",")
                if len(line_arr) < 2:
                    continue
                existing_ids[line_arr[0].strip("\s")] = line_arr[1]
                print(line)
        print("Getting " + str(len(existing_ids)) + " previous captions")
    except Exception:
        print("results file is new")

    # writing to csv file
    with open(results_file, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

        image_list = list(image_list.keys())
        print("working on ids from: " + str(args.min_image_id) + " to " + str(args.max_image_id))
        for n_im, impath in enumerate(image_list):
            if (n_im + 1) % 100 == 0:
                print("processing %d / %d" % (n_im + 1, len(image_list)))
            image_name = os.path.basename(impath)
            try:
                image_id = name_to_id_map[image_name]
            except Exception:
                #print("id error for: " + image_name)
                continue
            if int(image_id) < args.min_image_id or int(image_id) >= args.max_image_id:
                continue
            try:
                if image_id in existing_ids:
                    print("already have caption for id: " + str(image_id))
                    r = existing_ids[image_id]
                else:
                    r = get_caption(args, text_generator, impath)
                csvwriter.writerow([image_id, r])
                csvfile.flush()
            except Exception as er:
                print("CLIP error for" + image_name)
                print(er)
                continue
    print("All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--name_to_id_map", type=str, required=True, default="name_to_id_map.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--min_image_id", type=int, default=0)
    parser.add_argument("--max_image_id", type=int, default=500)

    args = parser.parse_args()

    extract_dataset_captions(args, args.image_dir, args.name_to_id_map, args.results_file)
