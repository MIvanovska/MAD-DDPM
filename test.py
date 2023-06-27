import argparse
from copy import deepcopy
import accelerate
import torch
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import transforms
from dataloader import ImageFolder
from tqdm.auto import tqdm
from evaluate import performances_compute
import k_diffusion as K
import os
from datetime import datetime
import json
from functools import partial

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--test-set', type=str, required=True, help='the testing set location')
    p.add_argument('--image_branch_checkpoint', type=str, required=True, help='path to the checkpoint of the image branch')
    p.add_argument('--features_branch_checkpoint', type=str, required=True, help='path to the checkpoint of the image branch')
    p.add_argument('--config', type=str, required=True, help='the configuration file')
    p.add_argument('--experiment_name', type=str, default='experiment', help='the name of the run')
    p.add_argument('--batch-size', type=int, default=4, help='the batch size')
    p.add_argument('--num-workers', type=int, default=8, help='the number of data loader workers')
    p.add_argument('--output_dir', type=str, default='./output', help='output directory fo trained models and results')

    args = p.parse_args()
    config = json.load(open(args.config))
    mp.set_start_method('spawn')
    now = datetime.now()
    output_log='./output/'+args.experiment_name + '_' + now.strftime("%d_%m_%Y__%H_%M_%S")+'.txt'


    assert len(config["model_image"]['input_size']) == 2 and config["model_image"]['input_size'][0] == config["model_image"]['input_size'][1]
    assert len(config["model_features"]['input_size']) == 2 and config["model_features"]['input_size'][0] == config["model_features"]['input_size'][1]
    img_size = config["model_image"]['input_size']
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size[0], img_size[1])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_set = ImageFolder(args.test_set, transform=tf, load_images=True, features=True)
    test_dl = data.DataLoader(test_set, args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, persistent_workers=True)
    ###########################################################################################
    # initialize the model
    inner_model_images = K.ImageDenoiserModelV1(c_in=config["model_image"]['input_channels'],
                                               feats_in=config["model_image"]['mapping_out'],
                                               depths=config["model_image"]['depths'],
                                               channels=config["model_image"]['channels'],
                                               self_attn_depths=config["model_image"]['self_attn_depths'],
                                               patch_size=config["model_image"]['patch_size'],
                                               dropout_rate=config["model_image"]['dropout_rate'])
    inner_model_features = K.ImageDenoiserModelV1(c_in=config["model_features"]['input_channels'],
                                               feats_in=config["model_features"]['mapping_out'],
                                               depths=config["model_features"]['depths'],
                                               channels=config["model_features"]['channels'],
                                               self_attn_depths=config["model_features"]['self_attn_depths'],
                                               patch_size=config["model_features"]['patch_size'],
                                               dropout_rate=config["model_features"]['dropout_rate'])


    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    print('Loading checkpoints {} and {}'.format(args.image_branch_checkpoint, args.features_branch_checkpoint))
    inner_model_images, inner_model_features, test_dl = accelerator.prepare(inner_model_images, inner_model_features, test_dl)
    model_image = K.Denoiser(inner_model_images, sigma_data=config["model_image"]['sigma_data'])
    ckpt = torch.load(args.image_branch_checkpoint, map_location='cpu')
    accelerator.unwrap_model(model_image.inner_model).load_state_dict(ckpt['model'])
    model_features = K.Denoiser(inner_model_features, sigma_data=config["model_features"]['sigma_data'])
    ckpt = torch.load(args.features_branch_checkpoint, map_location='cpu')
    accelerator.unwrap_model(model_features.inner_model).load_state_dict(ckpt['model'])
    del ckpt
    print('Checkpoints loaded successfully.')
    ###########################################################################################
    output_root = os.path.join(args.output_dir, args.experiment_name)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    sigma_min_image = config["model_image"]['sigma_min']
    sigma_min_features = config["model_features"]['sigma_min']

    with torch.no_grad():
        model_image.eval()
        model_features.eval()
        sigma_values = [0.5, 1.0, 1.5, 2.0]
        scores = torch.zeros(size=(len(sigma_values), len(test_dl.dataset),), dtype=torch.float32, device=device)
        gt_labels = torch.zeros(size=(len(test_dl.dataset),), dtype=torch.long, device=device)
        sample_names=[]
        for batch_id, batch in enumerate(tqdm(test_dl, disable=not accelerator.is_main_process)):
            reals_images = batch[0][0].to(device)
            reals_features = batch[0][1].to(device)
            reals_features = torch.reshape(reals_features, (reals_features.size(0), config["model_features"]['input_channels'], config["model_features"]['input_size'][0], config["model_features"]['input_size'][0]))
            gt_labels[batch_id * args.batch_size: batch_id * args.batch_size + batch[1].size(0)] = batch[1]
            filenames = batch[2]
            sample_names = sample_names + list(filenames)

            noise_images = torch.randn_like(reals_images).to(device)
            noise_features = torch.randn_like(reals_features).to(device)

            for sigma_id, value in enumerate(sigma_values):
                sigma = torch.FloatTensor([value]).to(device)

                x_images = model_image.add_noise(reals_images, noise_images, sigma)
                sigmas_images = K.utils.get_sigmas_karras(20, sigma_min_image, sigma.cpu(), rho=7., device=device)
                x_0_images = K.utils.sample_lms_test(model_image, x_images, sigmas_images, disable=not accelerator.is_main_process)
                x_0_images = accelerator.gather(x_0_images)[:args.batch_size]
                rec_images = (reals_images - x_0_images).view(x_0_images.size(0), x_0_images.size(1) * x_0_images.size(2) * x_0_images.size(3))
                error_images = torch.mean(torch.pow(rec_images, 2), dim=1)

                x_features = model_image.add_noise(reals_features, noise_features, sigma)
                sigmas_features = K.utils.get_sigmas_karras(20, sigma_min_features, sigma.cpu(), rho=7., device=device)
                x_0_features = K.utils.sample_lms_test(model_features, x_features, sigmas_features, disable=not accelerator.is_main_process)
                x_0_features = accelerator.gather(x_0_features)[:args.batch_size]
                rec_features = (reals_features - x_0_features).view(x_0_features.size(0), x_0_features.size(1) * x_0_features.size(2) * x_0_features.size(3))
                error_features = torch.mean(torch.pow(rec_features, 2), dim=1)

                error = error_images + error_features
                scores[sigma_id, batch_id * args.batch_size: batch_id * args.batch_size + error.size(0)] = error.reshape(error.size(0))

        labels = gt_labels.cpu()
        scores = scores.cpu()
        num_of_sigmas = len(sigma_values)
        val_auc = torch.zeros(size=(num_of_sigmas, ), dtype=torch.float32)
        val_eer = torch.zeros(size=(num_of_sigmas, ), dtype=torch.float32)
        threshold_APCER = torch.zeros(size=(num_of_sigmas, ), dtype=torch.float32)
        threshold_BPCER = torch.zeros(size=(num_of_sigmas, ), dtype=torch.float32)
        threshold_ACER = torch.zeros(size=(num_of_sigmas, ), dtype=torch.float32)
        with open(output_log, 'a') as f:
            for i in range(num_of_sigmas):
                f.write("Sigma: {}\n".format(sigma_values[i]))
                val_auc[i], val_eer[i], threshold_APCER[i], threshold_BPCER[i], threshold_ACER[i] = performances_compute(scores[i], labels, threshold_type='eer', op_val=0.1, verbose=False, positive_label=1)
                f.write("pos_label=1....AUC@ROC: {}, APCER:{}, EER: {}, BPCER:{}, ACER:{}\n".format(val_auc[i], threshold_APCER[i], val_eer[i], threshold_BPCER[i], threshold_ACER[i]))
                val_auc[i], val_eer[i], threshold_APCER[i], threshold_BPCER[i], threshold_ACER[i] = performances_compute(scores[i], labels, threshold_type='eer', op_val=0.1, verbose=False, positive_label=0)
                f.write("pos_label=0....AUC@ROC: {}, APCER:{}, EER: {}, BPCER:{}, ACER:{}\n".format(val_auc[i], threshold_APCER[i], val_eer[i], threshold_BPCER[i], threshold_ACER[i]))
        f.close()
    return None


if __name__ == '__main__':
    main()
