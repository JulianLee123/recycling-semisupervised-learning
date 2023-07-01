import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from data.crops import CropsDataSet
import glob

import pdb

import simsiam.loader
import simsiam.builder

from PIL import Image


from utils.yaml_config_hook import yaml_config_hook
import itertools
from itertools import cycle
import cv2
from matplotlib import gridspec
import math
import sys
import json

def inference(model, train_loader, human_image_list, subcat_name):
    metric_dict = {}
    similarity_f = nn.CosineSimilarity(dim=2)
    average_sim_list = []
    subcat_list = []
    with torch.no_grad():
        for step, (_, x_i1, x_i2, _, subcat_j , name_j, _) in enumerate(train_loader):
            x_j = human_image_list
            x_j = torch.permute(x_j, (0, 3, 1, 2))
            x_i = x_i1.to(args.device)
            x_j = x_j.to(args.device)
            # the paper takes the representations of the encoder and not the MLP during evaluation
            x = torch.cat((x_i, x_j))
            enc_opt = model(x)
            #p1, p2, z1, z2 = model(x, x) 
            # c = model.forward_cluster(x)
            sim = similarity_f(enc_opt.unsqueeze(1), enc_opt.unsqueeze(0))
            sim_viz = sim[0].cpu().detach().numpy()
            sims = sim[0][1:].cpu().detach().numpy()
            sims = sorted(sims)
            #top_5 = sims[-1]
            average_sim = sims[-1] # top 1 # np.average(sims)
            # average_sim = np.average(sims[-5:]) # top 5
            # average_sim = sims[-3] # 3rd best
            average_sim_list.append(average_sim)
            subcat_list.append(subcat_j[0])
            # print(subcat_j, average_sim)
       
            save_filename = "Average_Eval_viz/average_eval_"+str(step)+".jpg"
            #viz = visualize_similarity(sim_viz, x_i, x_j, save_filename, average_sim)
            # if step > 5:
            #     break

            # average_sim_list = np.array(average_sim_list)
        tp, fp, tn, fn, precision, recall, f1, threshold, subcat = calculate_metrics(average_sim_list, subcat_list, subcat_name)
        metric_dict["tp"] = tp
        metric_dict["fp"] = fp
        metric_dict["tn"] = tn
        metric_dict["fn"] = fn
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        metric_dict["f1"] = f1
        metric_dict["threshold"] = threshold
        metric_dict["subcat"] = subcat
        return metric_dict

def calculate_metrics(average_sim_list, subcat_list, subcat_name):
    best_f1 = -1
    best_threshold = -1
    for t in np.arange(0.1, 1.0, 0.02):
        # print("Threshold:", t)
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for (average_sim, subcat) in zip(average_sim_list, subcat_list):
            if subcat == subcat_name:
                if average_sim >= t:
                    tp+=1
                else:
                    fn+=1
            if subcat != subcat_name:
                if average_sim >= t:
                    fp+=1
                else:
                    tn+=1
        # if tp > 0 and fp > 0 and tn > 0 and fn > 0:
        precision = tp/(tp+fp)            
        recall = tp/(tp+fn)            
        f1 = (2 * precision * recall)/(precision + recall)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_tp = tp
            best_tn = tn
            best_fp = fp
            best_fn = fn
            best_f1 = f1
            best_threshold = t
    print("****************************")
    print("True Positives: ", best_tp)
    print("False Positives: ", best_fp)
    print("True Negatives", best_tn)
    print("False Negatives", best_fn)
    print("Precision: ", best_precision)
    print("Recall", best_recall)
    print("Best F1: ", best_f1, "Best Threshold: ", best_threshold)
    print("****************************")
    return best_tp, best_fp, best_tn, best_fn, best_precision, best_recall, best_f1, best_threshold, subcat_name

def plot_metrics(human_metric_dict, args):
    subcat_f1_mean_dict = {}
    subcat_f1_std_dict = {}
    subcats = ["Tins", "Crushed_Cans", "Intact_Cans",
           "Brown_Cardboard", "Coated_Cardboard", 
           "Trays", "Colored_Bottles",
           "Crushed_Bottles", "Intact_Bottles",
           "One_Gallon", "Half_Gallon"]
    for s in subcats:
        f1_list = []
        for filename, metrics in human_metric_dict.items():
            if s == metrics["subcat"]:
                f1_list.append(metrics["f1"])
        subcat_f1_mean_dict[s] = np.mean(f1_list)
        subcat_f1_std_dict[s] = np.std(f1_list)
    # print(subcat_metric_dict)

    plt.figure(figsize=[23, 15])
    col_map = plt.get_cmap('Paired')
    plt.bar(subcat_f1_mean_dict.keys(), subcat_f1_mean_dict.values(), width=0.5, edgecolor='y', 
        linewidth=3, yerr=subcat_f1_std_dict.values(), color=col_map.colors, ecolor='k', capsize=10)
    plt.grid(axis='y', color = 'olive', linestyle = '--')
    plt.xlabel('Categories', fontsize=15)
    plt.ylabel('F1 Score', fontsize=15)
    #plt.show()
    plt.savefig('save/eval_figs/' + str(args.run_name) + '.png')
    return subcat_f1_mean_dict, subcat_f1_std_dict



def visualize_similarity(sim_row, pos_crop, neg_crop, save_filename, average_sims):
    fig = plt.figure()
    neg_sim = sim_row[1:]
    plt.title("Average: " + str(average_sims), loc='left')
    outer_gs = gridspec.GridSpec(1, 2)
    inner_gs0 = outer_gs[0].subgridspec(1,1)
    axs_1 = fig.add_subplot(inner_gs0[0, 0])
    pos_sim = sim_row[0]
    pos_crop = torch.squeeze(pos_crop, 0)
    pos_crop = process_output_for_viz(pos_crop)
    pos_crop = cv2.cvtColor(pos_crop, cv2.COLOR_RGB2BGR)
    axs_1.imshow(pos_crop)
    axs_1.set_axis_off()
    axs_1.set_title(str(round(float(pos_sim), 3)))
            
    N = len(neg_sim)
    cols = 3
    rows = int(math.ceil(N / cols))
    for i, (nc, ns) in enumerate(zip(neg_crop, neg_sim)):
        gs = outer_gs[1].subgridspec(rows, cols)
        ax = fig.add_subplot(gs[i])
        nc = process_output_for_viz(nc)
        ax.set_axis_off()
        ax.imshow(nc)
        ax.set_title(str(round(float(ns), 3)))
    plt.axis('off')
    plt.savefig(save_filename)
    print(save_filename)
    return fig

def process_output_for_viz(crop_tensor):
    crop = crop_tensor.detach().cpu().numpy()
    crop = crop.transpose((1, 2, 0))
    crop = np.asarray(crop, np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    return crop
            

def main(gpu, args):
    #pdb.set_trace()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

     # initialize dataset
    if args.dataset == "crops":
        train_dataset = CropsDataSet(args.dataset_dir, 
                                    args.list_path, 
                                    args.mean,
                                    transforms=None)
        class_num = 11  

    train_sampler = None

    # initialize data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1, # so that crops are fed one by one to be compared to all the human crops
                                                   shuffle=(train_sampler is None),
                                                   drop_last=True,
                                                   num_workers=args.workers,
                                                   sampler=train_sampler,
                                                )

    args.run_name = "bs_" + str(args.batch_size) + "_projdim_" + str(args.pred_dim) + "_ep_" + str(args.epochs) + "_hlf_" + str(args.human_list_filename)
    if args.semi_supervised_weight != 0:
        args.run_name += "_ss_" + str(args.semi_supervised_weight)

    model_fp = os.path.join(args.model_path, args.dataset, args.run_name + "_" + str(args.eval_checkpoint_num) + ".tar")
    print(model_fp)

    # initialize full model (for debugging)
    # model_fp_full_tester = os.path.join(args.model_path, args.dataset, "checkpoint_100_4_12_old.tar".format(args.eval_epoch))
    # model_full = simsiam.builder.SimSiam(
    #     models.__dict__[args.arch],
    #     args.dim, args.pred_dim)
    # model_full = model_full.to('cuda')
    # model_full.eval()
    # model_full.load_state_dict(torch.load(model_fp_full_tester, map_location=args.device.type)['net'])   #['net']

    # initialize eval model (encoder from the trained model)
    model = models.__dict__[args.arch]()
    model = model.to('cuda')
    model.eval()
    checkpoint = torch.load(model_fp, map_location=args.device.type)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        #print(k)
        # retain only encoder up to before the embedding layer
        if k.startswith('encoder'):# and not k.startswith('encoder.fc'):
            # remove prefix
            state_dict[k[len("encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    #print('\n\n')
    #for k in list(state_dict.keys()):
    #    print(k)
    
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    
    human_metric_dict = {}
    data_root = os.path.join(os.path.expanduser("~"), "Crops_Dataset")
    human_text_file_root = os.path.join(data_root, 'file_lists/human_lists')
    for human_txt_file in sorted(os.listdir(human_text_file_root)):
    # for human_txt_file in sorted(glob.glob(human_text_file_root)):
        print(human_txt_file)
        human_data = np.loadtxt(os.path.join(human_text_file_root, human_txt_file), dtype=str)
        human_image_list = []
        human_name_list = []
        for cd in human_data:
            human_name_list.append(cd)
            image = Image.open(cd)
            image = image.resize((128, 128))
            image = np.asarray(image, np.float32)
            human_image_list.append(image)
            subcat = cd.split('/')[4]
        human_image_list = torch.as_tensor(human_image_list)

        metric_dict = inference(model, train_loader,  human_image_list, subcat)
        human_metric_dict[human_txt_file] = metric_dict
    
    # print(human_metric_dict)

    # plot_metrics(human_metric_dict, args)
    final_metric_dict = {}
    subcat_f1_mean_dict, subcat_f1_std_dict = plot_metrics(human_metric_dict, args)
    final_metric_dict["mean_f1"] = subcat_f1_mean_dict
    final_metric_dict["std_f1"] = subcat_f1_std_dict

    json_save_file = open(os.path.join("results/", str(args.run_name) + '.json'), "w")
    json.dump(final_metric_dict, json_save_file, indent=2)
    json_save_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simsiam")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = 1


    main(0, args)