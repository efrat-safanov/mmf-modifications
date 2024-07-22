#this splits images into multiple images based on uniform borders (up to 3)

#input - images folder

#output - per image that can be split split_#{orig_image_name}-00x.png


from PIL import Image
import os
from glob import glob
import csv
import argparse
import os


def split_image(args, img_path, id):
    img = Image.open(img_path)
    # get width and height
    width = img.width
    height = img.height

    min_size = int(args.size_ratio * width * height)
    command_args_default = " -u 3 -d " + str(min_size) + " -S 1 -t 4 "
    command_args_mid_pic_east = " -u 3 -d " + str(min_size) + "  -c East -S 1 -t 4 "
    command_args_mid_pic_west = " -u 3 -d " + str(min_size) + "  -c West -S 1 -t 4 "
    command_args_mid_pic_south = " -u 3 -d " + str(min_size) + "  -c South -S 1 -t 4 "
    command_args_black = " -u 3 -d " + str(min_size) + " -b \"rgb(0,0,0)\" -S 1 -t 4 "

    new_image_path = args.out_dir + "/" + str(id) + ".png"
    commands = [command_args_default, command_args_mid_pic_east, command_args_mid_pic_west, command_args_mid_pic_south, command_args_black]

    for c in commands:
        full_cmd = "./multicrop2.sh" + " " + c + " " + img_path + " " + new_image_path
        print(full_cmd)
        res = os.popen(full_cmd).read()
        if "Processing Image 1" in res:
            print("got split")
            return True
    os.popen("rm " + args.out_dir + "/" + str(id) + "-000.png")
    return False


def split_dataset_images(args, image_dir, name_to_id_map_file):
    image_list = glob(image_dir + "/*.*")
    splits_done = 0
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

    image_list = list(image_list.keys())
    for n_im, impath in enumerate(image_list):
        if (n_im + 1) % 100 == 0:
            print("processing %d / %d" % (n_im + 1, len(image_list)))
        image_name = os.path.basename(impath)
        try:
            image_id = name_to_id_map[image_name]
        except Exception:
            #print("id error for: " + image_name)
            continue
        try:
            if split_image(args, impath, image_id):
                splits_done += 1
        except Exception as er:
            print("Split error for" + image_name)
            print(er)
            continue
    print("All done! " + str(splits_done) + " splits done overall")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--name_to_id_map", type=str, required=True, default="name_to_id_map.csv")
    parser.add_argument("--size_ratio", type=float, default=0.25)
    args = parser.parse_args()

    split_dataset_images(args, args.image_dir, args.name_to_id_map)
