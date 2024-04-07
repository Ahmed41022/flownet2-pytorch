import argparse
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb
from datasets import *
model_names = ['FlowNet2']


parser = argparse.ArgumentParser(
    description="PyTorch FlowNet2 inference on a folder of img pairs",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "data",
    metavar="DIR",
    help="path to images folder, image names must match '[name]0.[ext]' and '[name]1.[ext]'",
)
parser.add_argument("pretrained", metavar="PTH",
                    help="path to pre-trained model")
parser.add_argument(
    "--output",
    "-o",
    metavar="DIR",
    default=None,
    help="path to output folder. If not set, will be created in data folder",
)
parser.add_argument(
    "--output-value",
    "-v",
    choices=["raw", "vis", "both"],
    default="both",
    help="which value to output, between raw input (as a npy file) and color vizualisation (as an image file)."
    " If not set, will output both",
)
parser.add_argument(
    "--div-flow",
    default=20,
    type=float,
    help="value by which flow will be divided. overwritten if stored in pretrained file",
)
parser.add_argument(
    "--img-exts",
    metavar="EXT",
    default=["png", "jpg", "bmp", "ppm"],
    nargs="*",
    type=str,
    help="images extensions to glob",
)
parser.add_argument(
    "--max_flow",
    default=None,
    type=float,
    help="max flow value. Flow map color is saturated above this value. If not set, will use flow map's max value",
)
parser.add_argument(
    "--upsampling",
    "-u",
    choices=["nearest", "bilinear"],
    default=None,
    help="if not set, will output FlowNet raw input,"
    "which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling",
)
parser.add_argument(
    "--bidirectional",
    action="store_true",
    help="if set, will output invert flow (from 1 to 0) along with regular flow",
)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    if args.output_value == "both":
        output_string = "raw output and RGB visualization"
    elif args.output_value == "raw":
        output_string = "raw output"
    elif args.output_value == "vis":
        output_string = "RGB visualization"
    print("=> will save " + output_string)
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir / "flow"
    else:
        save_path = Path(args.output)
    print("=> will save everything to {}".format(save_path))
    save_path.makedirs_p()
    # Data loading code
    input_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.glob(
            "frame[0-9][0-9][0-9]." + ext)  # Select all frames
        # Sort the files to ensure the correct order
        test_files = sorted(test_files)

        # Iterate through the files, excluding the last one
        for i in range(len(test_files) - 1):
            file1 = test_files[i]
            file2 = test_files[i + 1]  # Get the next file

            img_pair = (file1, file2)
            if file2.isfile():
                img_pairs.append(img_pair)

    # Reset img_pairs to an empty list after the loop, as per your request

    print("{} samples found".format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model FlowNet2")
    model = models.FlowNet2(args)
    model.load_state_dict(network_data['state_dict'], strict=False)

    model.to(device)
    model.eval()
    cudnn.benchmark = True

    if "div_flow" in network_data.keys():
        args.div_flow = network_data["div_flow"]

    for img1_file, img2_file in tqdm(img_pairs):

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        images = [img1, img2]
        images = np.array(images).transpose(3, 0, 1, 2)
        input_var = torch.from_numpy(images.astype(np.float32))
        input_var = input_var.unsqueeze(0)  # Add a batch dimension
        if args.bidirectional:
            # feed inverted pair along with normal pair
            inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
            input_var = torch.cat([input_var, inverted_input_var])

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)
        if args.upsampling is not None:
            output = F.interpolate(
                output, size=img1.size(
                )[-2:], mode=args.upsampling, align_corners=False
            )
        for suffix, flow_output in zip(["flow", "inv_flow"], output):
            filename = save_path / "{}{}".format(img1_file.stem[:-1], suffix)
            if args.output_value in ["vis", "both"]:
                rgb_flow = flow2rgb(
                    args.div_flow * flow_output, max_value=args.max_flow
                )
                to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
                imwrite(filename + ".png", to_save)
            if args.output_value in ["raw", "both"]:
                # Make the flow map a HxWx2 array as in .flo files
                to_save = (args.div_flow *
                           flow_output).cpu().numpy().transpose(1, 2, 0)
                np.save(filename + ".npy", to_save)


if __name__ == "__main__":
    main()
