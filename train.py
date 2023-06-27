import argparse
from copy import deepcopy
import accelerate
import torch
from torch import optim
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
    p.add_argument('--train-set', type=str, required=True, help='the training set location')
    p.add_argument('--config', type=str, required=True, help='the configuration file')
    p.add_argument('--branch', type=int, required=True, default=0, help='choose the branch you want to train or test: 1-image and 2-features')
    p.add_argument('--experiment_name', type=str, default='experiment', help='the name of the run')
    p.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    p.add_argument('--batch-size', type=int, default=1, help='the batch size')
    p.add_argument('--num-workers', type=int, default=8, help='the number of data loader workers')
    p.add_argument('--output_dir', type=str, default='./output', help='output directory fo trained models and results')
    p.add_argument('--save-every', type=int, default=10, help='save every this many epochs')
    p.add_argument('--resume', type=str, help='path to the checkpoint to resume from')
    p.add_argument('--test-every', type=int, default=-1, help='evaluate model every this many epochs (value -1 is used for training without evaluation)')
    p.add_argument('--test-set', type=str, default="", help='the testing set location, if you perform evaluation while training')

    args = p.parse_args()
    config = json.load(open(args.config))
    mp.set_start_method('spawn')
    now = datetime.now()
    output_log='./output/'+args.experiment_name + '_' + now.strftime("%d_%m_%Y__%H_%M_%S")+'.txt'

    assert args.branch==1 or args.branch==2
    if args.branch==1:
        branch="model_image"
        assert len(config[branch]['input_size']) == 2 and config[branch]['input_size'][0] == config[branch]['input_size'][1]
        print("Input sample size: {} x {}, number of channels: {}".format(config[branch]['input_size'][0], config[branch]['input_size'][1], config[branch]['input_channels']))
        img_size = config[branch]['input_size']
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        features_flag = False
        images_flag = True
    if args.branch==2:
        branch = "model_features"
        assert len(config[branch]['input_size']) == 2 and config[branch]['input_size'][0] == config[branch]['input_size'][1]
        tf = None
        features_flag = True
        images_flag = False
    ###########################################################################################
    # initialize the model
    inner_model = K.ImageDenoiserModelV1(c_in=config[branch]['input_channels'],
                                               feats_in=config[branch]['mapping_out'],
                                               depths=config[branch]['depths'],
                                               channels=config[branch]['channels'],
                                               self_attn_depths=config[branch]['self_attn_depths'],
                                               patch_size=config[branch]['patch_size'],
                                               dropout_rate=config[branch]['dropout_rate'])
    print('Number of parameters in the model:', K.utils.n_params(inner_model))
    opt = optim.AdamW(inner_model.parameters(), lr=config['optimizer']['lr'], betas=tuple(config['optimizer']['betas']), eps=config['optimizer']['eps'],
                      weight_decay=config['optimizer']['weight_decay'])
    sched = K.utils.InverseLR(opt, inv_gamma=config['lr_sched']['inv_gamma'], power=config['lr_sched']['power'], warmup=config['lr_sched']['warmup'])
    ema_sched = K.utils.EMAWarmup(power=config['ema_sched']['power'], max_value=config['ema_sched']['max_value'])

    ###########################################################################################
    # initialize the dataloaders
    train_set = ImageFolder(args.train_set, transform=tf, load_images=images_flag, features=features_flag)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=False, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True)
    if args.test_every !=-1:
        assert args.test_set != ""
        test_set = ImageFolder(args.test_set, transform=tf, load_images=images_flag, features=features_flag)
        test_dl = data.DataLoader(test_set, args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=args.num_workers, persistent_workers=True)

    ###########################################################################################
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    inner_model, opt, train_dl = accelerator.prepare(inner_model, opt, train_dl)

    model = K.Denoiser(inner_model, sigma_data=config[branch]['sigma_data'])
    model_ema = deepcopy(model)
    if args.resume:
        print('Loading checkpoint ', args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        accelerator.unwrap_model(model.inner_model).load_state_dict(ckpt['model'])
        accelerator.unwrap_model(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        del ckpt
        print('Checkpoint loaded successfully.')

    else:
        epoch = 0
        step = 0

    output_root = os.path.join(args.output_dir, args.experiment_name)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    sigma_min = config[branch]['sigma_min']
    # sample_density = partial(K.utils.rand_log_normal, loc=config[branch]['mean'], scale=config[branch]['std']) # lognormal
    sample_density = partial(K.utils.rand_log_logistic, loc=config[branch]['mean'], scale=config[branch]['std'], min_value=config[branch]['sigma_min'], max_value=config[branch]['sigma_max'])

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def save_checkpoint():
        accelerator.wait_for_everyone()
        if not os.path.isdir(os.path.join(output_root, "checkpoints")):
            os.makedirs(os.path.join(output_root, "checkpoints"))
        filename = f'{os.path.join(output_root, "checkpoints", branch+"_"+args.experiment_name)}_epoch_{epoch:03}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model.inner_model).state_dict(),
            'model_ema': accelerator.unwrap_model(model_ema.inner_model).state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': None,
        }
        accelerator.save(obj, filename)

    def test(test_dl, epoch, features_flag):
        with torch.no_grad():
            model.eval()
            sigma_values = [0.5, 1.0, 1.5, 2.0]
            scores = torch.zeros(size=(len(sigma_values), len(test_dl.dataset),), dtype=torch.float32, device=device)
            gt_labels = torch.zeros(size=(len(test_dl.dataset),), dtype=torch.long, device=device)
            image_names=[]
            for batch_id, batch in enumerate(tqdm(test_dl, disable=not accelerator.is_main_process)):
                reals = batch[0].to(device)
                if features_flag:
                    reals = torch.reshape(reals, (reals.size(0), config[branch]['input_channels'], config[branch]['input_size'][0], config[branch]['input_size'][0]))
                gt_labels[batch_id * args.batch_size: batch_id * args.batch_size + batch[1].size(0)] = batch[1]
                filenames = batch[2]
                image_names = image_names + list(filenames)
                noise = torch.randn_like(reals).to(device)

                for sigma_id, value in enumerate(sigma_values):
                    sigma = torch.FloatTensor([value]).to(device)
                    x = model.add_noise(reals, noise, sigma)
                    sigmas = K.utils.get_sigmas_karras(20, sigma_min, sigma.cpu(), rho=7., device=device)
                    x_0 = K.utils.sample_lms_test(model, x, sigmas, disable=not accelerator.is_main_process)
                    x_0 = accelerator.gather(x_0)[:args.batch_size]
                    rec = (reals - x_0).view(x_0.size(0), x_0.size(1) * x_0.size(2) * x_0.size(3))
                    error = torch.mean(torch.pow(rec, 2), dim=1)
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
                f.write("Epoch: {}\n".format(epoch))
                for i in range(num_of_sigmas):
                    f.write("Sigma: {}\n".format(sigma_values[i]))
                    val_auc[i], val_eer[i], threshold_APCER[i], threshold_BPCER[i], threshold_ACER[i] = performances_compute(scores[i], labels, threshold_type='eer', op_val=0.1, verbose=False, positive_label=1)
                    f.write("pos_label=1....AUC@ROC: {}, APCER:{}, EER: {}, BPCER:{}, ACER:{}\n".format(val_auc[i], threshold_APCER[i], val_eer[i], threshold_BPCER[i], threshold_ACER[i]))
                    val_auc[i], val_eer[i], threshold_APCER[i], threshold_BPCER[i], threshold_ACER[i] = performances_compute(scores[i], labels, threshold_type='eer', op_val=0.1, verbose=False, positive_label=0)
                    f.write("pos_label=0....AUC@ROC: {}, APCER:{}, EER: {}, BPCER:{}, ACER:{}\n".format(val_auc[i], threshold_APCER[i], val_eer[i], threshold_BPCER[i], threshold_ACER[i]))
            f.close()
            model.train()
        return None

    try:
        while epoch<args.num_epochs:
            model.train()
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                opt.zero_grad()
                reals = batch[0].to(device)
                if features_flag:
                    reals = torch.reshape(reals, (reals.size(0), config[branch]['input_channels'], config[branch]['input_size'][0], config[branch]['input_size'][0]))
                noise = torch.randn_like(reals)
                sigma = sample_density([reals.shape[0]], device=device)
                losses = model.loss(reals, noise, sigma)
                losses_all = accelerator.gather(losses.detach())
                loss_local = losses.mean()
                loss = losses_all.mean()
                accelerator.backward(loss_local)
                opt.step()
                sched.step()
                ema_decay = ema_sched.get_value()
                K.utils.ema_update(model, model_ema, ema_decay)
                ema_sched.step()
                step += 1
            if epoch !=0:
                if epoch % args.save_every == 0:
                    save_checkpoint()
                if args.test_every != -1 and epoch % args.test_every == 0:
                    test(test_dl, epoch, features_flag)
            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')
            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
