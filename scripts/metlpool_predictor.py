
import os
import sys
sys.path.append('./notebooks/')


import torch
import numpy as np
import matplotlib.pyplot as plt
# Turn on interactive mode
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import pandas as pd
import torch
import json
from ops.utils import*
from matplotlib.colors import ListedColormap
import os
import shutil 
from sam2.build_sam import build_sam2_video_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



# CUDA_VISIBLE_DEVICES=0 python /home/mohamed/data2/projects/segment-anything-2/scripts/metlpool_predictor.py
# CUDA_VISIBLE_DEVICES=1 python /home/mohamed/data2/projects/segment-anything-2/scripts/metlpool_predictor.py
num_frames = 200
skiprate = 1 
_unique_id='v4_0i2sz'

num_frames = 200
skiprate = 75 
_unique_id='v5_qew(d312'

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
fill_hole_area = 0


masks_directory = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240712/CRED/20240708_09_00/CREDgt_set_4_v3/CREDgt_set_4/Masks/"
raw_video_dir = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240712/CRED/20240708_09_00/20240708_09_00_12072024_113101.raw"
################

masks_directory = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240801/20240801gt_set_5/20240801gt_set_5/Masks"
raw_video_dir = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240801/20240713_20_00_01082024_145636.raw"

# ################

################
masks_directory = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240611/20240604_13_00/CREDgt_set_1/CREDgt_set_1/Masks"
raw_video_dir = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240611/20240604_13_00/20240604_13_00_11062024_135808.raw"
################
# masks_directory = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240626/20240626_03_00/CREDgt_set_2/Masks"
# raw_video_dir = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240626/20240626_03_00/20240626_03_00_26062024_155807.raw"

# ###############
# masks_directory = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240711/20240708_07_02/CREDgt_set_3_v2/Mask"
# raw_video_dir = "/mnt/md126/users/mohamed/projects/AM/Data/RAW/ByDay/20240711/20240708_07_02/20240708_07_02_11072024_175349.raw"
# ###############


# Temporary directory to save reordered frames
temp_dir = raw_video_dir.replace("/Data/RAW/ByDay/", f"/Data/Processed/Noisy_Pseudo_Masks/{_unique_id}/").replace(".raw", "/")
# temp_dir = f"./videos/{data_set_tag}/"
orginal_dir = f"{temp_dir}/orginal/"
sam_dir = f"{temp_dir}/sam/"
sam_temp_dir = f"{sam_dir}/temp/"
sam_results_dir = f"{sam_dir}/results/"
xinyue_dir = f"{temp_dir}/xinyue/"




data_set_tag = f'noisy_unique_id{_unique_id}'
flood_masks = sort_and_filter_images(masks_directory, "_c1")   # Sort and filter the images
voids_masks = sort_and_filter_images(masks_directory, "_c2")   # Sort and filter the images
masks_frames = [int(name.split('_f_')[1].split('_')[0]) for name in flood_masks]


slected_mask_index = 5 # the index is the order of the mask in the list of masks xinyue has labeled.
for slected_mask_index in range(len(masks_frames)):
    mask_frame = masks_frames[slected_mask_index] # index based camera frame number
    flood_mask_path = flood_masks[slected_mask_index]
    voids_mask_list = [f for f in voids_masks if f"f_{mask_frame}_c2" in f]
    if len(voids_mask_list) == 0:
        voids_mask_path = None
        
        print(f"Voids mask not found for {mask_frame}")
        print(f'flood_mask_path: {masks_directory}/{flood_mask_path}')
    elif len(voids_mask_list) == 1:
        voids_mask_path = voids_mask_list[0]
    else:
        raise ValueError(f"Multiple voids masks found for {mask_frame}") # For cases where Xinque has not labeled the voids because the meltpool did not have any voids


    parameters = {
        "mask_frame_used": slected_mask_index,
        "model_name": "SAM 2",
        "raw_file_directory": raw_video_dir,
        "masks_directory": masks_directory,
        "num_frames": num_frames,
        "skiprate": skiprate,
        "unique_id": _unique_id,
        "temp_dir": temp_dir,
        "orginal_dir": orginal_dir,
        "sam_temp_dir": sam_temp_dir,
        "sam_results_dir": sam_results_dir,
        "xinyue_dir": xinyue_dir,
        'sam2_checkpoint': sam2_checkpoint,
        'model_cfg': model_cfg,
        'fill_hole_area':fill_hole_area,
    }

    # shutil.rmtree(temp_dir, ignore_errors=True)  # Remove the directory if it already exists
    shutil.rmtree(sam_temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True) 
    os.makedirs(xinyue_dir, exist_ok=True) 
    # os.makedirs(orginal_dir, exist_ok=True) 
    os.makedirs(sam_temp_dir, exist_ok=True)
    os.makedirs(sam_results_dir, exist_ok=True)
    # os.makedirs(f"{sam_results_dir}/masks/", exist_ok=True) 
    # os.makedirs(f"{sam_results_dir}/overlay/", exist_ok=True) 


    frames_indices = [ mask_frame ]

    for i in range(num_frames):
        frames_indices.append(mask_frame + i*skiprate) # index based camera frame number
    frames_indices = list(set(frames_indices)) 
    frames_indices.sort()
    mask_index = frames_indices.index(mask_frame) # sam index
    image_meta_data = {'filename_abs': raw_video_dir, 'height': 512, 'width': 640}
    mask_frame_array = open_frame_firstlight(image_meta_data, mask_frame)['img']
    pixel_wise_diff_list = []
    missing_frames = []
    for i in frames_indices: # index based camera frame number
        try:
            frame = open_frame_firstlight(image_meta_data, i)
        except:
            print(f"Frame {i} not found")
            missing_frames.append(i)
            continue
        img_array = frame['img']
        pixel_wise_diff = np.linalg.norm(img_array - mask_frame_array)
        pixel_wise_diff_list.append(pixel_wise_diff)
        img = Image.fromarray(img_array)
        img = img.convert('L')
        # img.save(f"{orginal_dir}/{i}.jpg") 
        img.save(f"{sam_temp_dir}/{frames_indices.index(i)}.jpg") # sam index 1,2,3,4,5,6,7,8,9,10,..........
    parameters['frames_indices'] = frames_indices


    inference_state = predictor.init_state(video_path=sam_temp_dir)
    
    predictor.reset_state(inference_state)
    predictor.fill_hole_area = fill_hole_area

    # Load the image and masks
    image = Image.fromarray(mask_frame_array) 
    flood_mask = Image.open(os.path.join(masks_directory, flood_mask_path))
    voids_mask = Image.open(os.path.join(masks_directory, voids_mask_path)) if voids_mask_path is not None else None
    # Convert masks to binary arrays
    flood_mask_binary = np.array(flood_mask)[:, :, 0] > 250  # Extract red channel
    if voids_mask is None: # For cases where Xinque has not labeled the voids because the meltpool did not have any voids
        void_mask_binary = (np.array(flood_mask)*0)[:, :, 0]  > 250
        voids_mask = Image.fromarray(np.zeros(flood_mask.size[::-1], dtype=np.uint8))
    else:
        void_mask_binary = np.array(voids_mask)[:, :, 1] > 250  # Extract green channel


    

    # Create the meltpool and background masks
    meltpool_mask = flood_mask_binary & ~void_mask_binary
    background_mask = ~meltpool_mask

    # saving the masks
    void_mask_binary_img = Image.fromarray(void_mask_binary.astype(np.uint8) * 255)
    flood_mask_binary_img = Image.fromarray(flood_mask_binary.astype(np.uint8) * 255)
    meltpool_mask_img = Image.fromarray(meltpool_mask.astype(np.uint8) * 255)
    background_mask_img = Image.fromarray(background_mask.astype(np.uint8) * 255)

    # flood_mask_binary_img.save(f"{xinyue_dir}/flood_mask_binary.jpg")
    # void_mask_binary_img.save(f"{xinyue_dir}/void_mask_binary.jpg")
    # meltpool_mask_img.save(f"{xinyue_dir}/meltpool_mask.jpg")
    # background_mask_img.save(f"{xinyue_dir}/background_mask.jpg")
    # img.save(f"{xinyue_dir}/original.jpg")



    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # sample_points_p = sample_points(meltpool_mask, 10)
    # sample_points_n = sample_points(background_mask, 10)
    # sample_points_n_2 = sample_points(void_mask_binary, 10) 
    # points = np.concatenate([sample_points_p, sample_points_n, sample_points_n_2], axis=0)
    # labels = np.concatenate([np.ones(len(sample_points_p)), np.zeros(len(sample_points_n)), np.zeros(len(sample_points_n_2))], axis=0)

    # for labels, `1` means positive click and `0` means negative click
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points(
    #     inference_state=inference_state,
    #     frame_idx=frame_idx-1,
    #     obj_id=ann_obj_id,
    #     points=points,
    #     labels=labels,
    # )
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=mask_index, # sam index
        obj_id=ann_obj_id,
        mask=meltpool_mask,
    )

    # Create a figure with 4x2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(16, 24))
    # Display images and masks


    axs[0, 1].imshow(flood_mask, cmap="gray")
    axs[0, 1].set_title("Xinyue Flood Mask")
    axs[0, 1].axis('off')

    axs[1, 1].imshow(flood_mask_binary, cmap='gray')
    axs[1, 1].set_title('Binarized Xinyue Flood Mask')
    axs[1, 1].axis('off')

    axs[0, 0].imshow(voids_mask, cmap="gray")
    axs[0, 0].set_title("Xinyue Voids Mask")
    axs[0, 0].axis('off')


    axs[1, 0].imshow(void_mask_binary, cmap='gray')
    axs[1, 0].set_title('Binarized Xinyue Voids Mask')
    axs[1, 0].axis('off')

    axs[2, 1].imshow(meltpool_mask, cmap='gray')
    axs[2, 1].set_title('Prompt')
    axs[2, 1].axis('off')


    axs[2, 0].imshow(image, cmap="gray")
    axs[2, 0].set_title(f"Original Frame {mask_index}")
    axs[2, 0].axis('off')
    axs[3, 0].imshow(image)
    single_color_cmap_blue = ListedColormap(['none', 'blue']) 
    SAM2_Mask= (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
    axs[3, 0].imshow(SAM2_Mask, alpha=0.5,cmap=single_color_cmap_blue)  
    axs[3, 0].set_title('SAM2 Meltpool Mask Mask')
    axs[3, 0].axis('off')

    # Overlay meltpool mask on the original image
    single_color_cmap_red = ListedColormap(['none', 'red']) 
    axs[3, 1].imshow(image, cmap='gray')
    axs[3, 1].imshow(meltpool_mask, alpha=0.5,cmap=single_color_cmap_red)  
    axs[3, 1].set_title('Xinque Meltpool Mask')
    axs[3, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{xinyue_dir}/all_masks_{mask_frame}.jpg")
    plt.show()
    plt.close()


    # Assuming out_mask_logits are raw logits, and assuming we are only intrested in single object, we convert them to probabilities for confidence calculation
    sigmoid = torch.nn.Sigmoid()
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    frame_metrics = {}  # mask_confidence contains the per-frame mask confidence scores
    frame_metrics['mask_confidence'] = {}
    frame_metrics['frame'] = {}
    frame_metrics['mask_area'] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # out_frame_idx is according sam index
            mask = (out_mask_logits[ann_obj_id-1] > 0.0).cpu().numpy() #
            video_segments[out_frame_idx] = mask
            # Convert logits to probabilities for the current mask
            probabilities = sigmoid(out_mask_logits[ann_obj_id-1]).cpu().numpy()
            frame_metrics['mask_confidence'][out_frame_idx] = probabilities[mask].mean()  # Mean confidence of the mask
            frame_metrics['frame'] [out_frame_idx] = frames_indices[out_frame_idx]
            frame_metrics['mask_area'][out_frame_idx] = mask.sum() 

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(sam_temp_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))



    vis_dpi = 1200  
    single_color_cmap_blue = ListedColormap(['none', 'blue'])
    width = image_meta_data['width']
    height = image_meta_data['height']

    parameters['vis_dpi'] = vis_dpi
    parameters['width'] = width
    parameters['height'] = height

    # Calculate figure size in inches
    fig_width = width / vis_dpi
    fig_height = height / vis_dpi
    max_digits = len(str(999999999999))
    plt.close("all")
    frame_metrics['file_name'] = {}
    for out_frame_idx in range(0, len(frame_names), 1):
        frame_number = frames_indices[out_frame_idx]
        if frame_number in missing_frames:
            continue
        
        # check if the frame is already exported 
        previous_masks_same_frame = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(sam_results_dir) if f.startswith(f"masks_{frame_number}")]
        if len(previous_masks_same_frame) > 0:
            last_counter = max(previous_masks_same_frame)
            mask_counter = last_counter + 1
        else: 
            mask_counter = 0
        mask_counter_str = str(mask_counter).zfill(max_digits)
        frame_number_str = str(frame_number).zfill(max_digits)
        frame_metrics['file_name'][out_frame_idx] = f"{frame_number_str}_{mask_counter_str}"
            # find all frames that has 
        # Overlay the segmentation mask
        plt.figure(figsize=(fig_width, fig_height), dpi=vis_dpi)
        plt.imshow(Image.open(os.path.join(sam_temp_dir, frame_names[out_frame_idx])), cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)  # Set padding to zero for tight layout
        plt.subplots_adjust(wspace=0, hspace=0)  # Remove any space between subplots
        out_mask = video_segments[out_frame_idx]
        plt.imshow(out_mask.squeeze(), alpha=0.25, cmap=single_color_cmap_blue)
        out_frame_idx_str = str(out_frame_idx).zfill(max_digits)
        plt.savefig(f"{sam_results_dir}/overlay_{frame_number_str}_{mask_counter_str}.png")
        plt.close("all")
        
        # Save the mask alone
        plt.figure(figsize=(fig_width, fig_height), dpi=vis_dpi)
        plt.imshow(out_mask.squeeze(), alpha=1, cmap=ListedColormap(['black', 'white']))
        plt.axis('off')
        plt.tight_layout(pad=0)  # Set padding to zero for tight layout
        plt.subplots_adjust(wspace=0, hspace=0)  # Remove any space between subplots
        plt.savefig(f"{sam_results_dir}/masks_{frame_number_str}_{mask_counter_str}.png")
        plt.close("all")

        # Save the orginal image
        plt.figure(figsize=(fig_width, fig_height), dpi=vis_dpi)
        plt.imshow(Image.open(os.path.join(sam_temp_dir, frame_names[out_frame_idx])), cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)  # Set padding to zero for tight layout
        plt.subplots_adjust(wspace=0, hspace=0)  # Remove any space between subplots
        plt.savefig(f"{sam_results_dir}/input_{frame_number_str}_{mask_counter_str}.png")
        plt.close("all")


    # Create a DataFrame to store the mask confidence scores from the dictionary
    mask_confidence_df = pd.DataFrame(frame_metrics)
    mask_confidence_df['pixel_wise_diff'] = pixel_wise_diff_list


    csv_file_dir = f"{sam_dir}/mask_confidence_{mask_frame}.csv"
    parameters['csv_file_dir'] = csv_file_dir
    mask_confidence_df.to_csv(csv_file_dir, index=False)

    json_file_path = os.path.join(sam_dir, f"parameters_{mask_frame}.json")  # Saving in the same directory as your CSV
    with open(json_file_path, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)
    print(f"Parameters saved to {json_file_path}")

    shutil.rmtree(sam_temp_dir, ignore_errors=True)

# Cobine all the csv files into one
csv_files = [f for f in os.listdir(sam_dir) if f.startswith("mask_confidence_")]
combined_csv = pd.concat([pd.read_csv(f"{sam_dir}/{f}") for f in csv_files])
combined_csv['Normalized_Pixel_Wise_Diff'] = combined_csv['pixel_wise_diff'] / combined_csv['pixel_wise_diff'].max()
combined_csv.to_csv(f"{sam_dir}/combined_mask_confidence.csv", index=False)




