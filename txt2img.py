# other scripts this is based on
# https://github.com/CompVis/stable-diffusion - original colab version
# https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb - code to use k_lms
# DrKoin worked out the code for saving the seeds with inidivdual images

import sys

sys.stdout.write("Imports ...\n")
sys.stdout.flush()

sys.path.append('./k-diffusion')

import os
os.environ["XDG_CACHE_HOME"] = "../../.cache"

import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import k_diffusion as K
import accelerate


sys.stdout.write("Parsing arguments ...\n")
sys.stdout.flush()

def parse_args():
    desc = "Blah"

    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, help="the prompt to render")
    parser.add_argument("--init_img", type=str, help="path to the input image")
    parser.add_argument("--strength", type=float, help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image")
    parser.add_argument("--outdir", type=str, default="outputs/txt2img-samples", help="dir to write results to")
    parser.add_argument("--skip_grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples")
    parser.add_argument("--skip_save", action='store_true', help="do not save individual samples. For speed measurements.")
    parser.add_argument("--ddim_steps", type=int, help="number of ddim sampling steps")
    parser.add_argument("--sampler", type=str, help="plms, ddim, k_lms, etc")
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across samples ")
    parser.add_argument("--ddim_eta", type=float, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--n_iter", type=int, help="sample this often")
    parser.add_argument("--H", type=int, help="image height, in pixel space")
    parser.add_argument("--W", type=int, help="image width, in pixel space")
    parser.add_argument("--C", type=int, help="latent channels")
    parser.add_argument("--f", type=int, help="downsampling factor")
    parser.add_argument("--n_samples", type=int, help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--n_rows", type=int, help="rows in the grid (default: n_samples)")
    parser.add_argument("--scale",  type=float, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint of model")
    parser.add_argument("--seed", type=int, help="the seed (for reproducible sampling)")
    parser.add_argument("--precision", type=str, help="evaluate at this precision")
    parser.add_argument("--image_file", type=str)
    parser.add_argument("--dynamic_threshold",  type=float, help="?")
    parser.add_argument("--static_threshold",  type=float, help="?")
    parser.add_argument("--embedding_path", type=str, help="Path to a pre-trained embedding manager checkpoint")

    args = parser.parse_args()
    return args

args=parse_args();

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print('Using device:', device)
print(torch.cuda.get_device_properties(device))
sys.stdout.flush()

"""
def split_weighted_subprompts(text):
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight 
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights 
"""

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    #w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    #image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = image.resize((args.W, args.H), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def make_callback(sampler, dynamic_threshold=None, static_threshold=None):  
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        
        sys.stdout.write(f"Iteration {args_dict['i']+1}\n")
        sys.stdout.flush()
        
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        
        sys.stdout.write(f"Iteration {i+1}\n")
        sys.stdout.flush()
        
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1*static_threshold, static_threshold)

    if sampler in ["plms","ddim"]: 
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else: 
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback

def create_tensors(shape, seeds, step, passed_sampler=None, passed_t_enc=None):

    #Create the tensors and pass them to the scripts afterwards
    if args.init_img is None:
        img = torch.randn([1, *shape], device=device)
    else:
        img = passed_sampler.stochastic_encode(shape, torch.tensor([passed_t_enc]).to(device))

    if(args.n_samples==1):
        print(f"\nStarting batch step {step}/{args.n_iter}. 1 image using the following seed : {seeds[0]}\n")
        return img
    
    #print() or sys.stdout.write()? Unclear about the difference and "better" code
    sys.stdout.write(f"\nStarting batch step {step}/{args.n_iter}. {args.n_samples} images using the following seeds :\n")
    sys.stdout.write(' '.join(str(item) for item in seeds) + '\n')
    sys.stdout.flush()
    
    for loop in range(1, args.n_samples):
        torch.manual_seed(seeds[loop])
        if args.init_img is None:
            newsample = torch.randn([1, *shape], device=device)
        else:
            newsample = passed_sampler.stochastic_encode(shape, torch.tensor([passed_t_enc]).to(device))
        img = torch.cat((img, newsample))
    return img        
    
def get_unique_filename(file_name):
    new_file_name = file_name
    if os.path.isfile(file_name):
        expand = 0
        while os.path.isfile(new_file_name):
            expand += 1
            #new_file_name = file_name.split(".png")[0] + " "+str(expand) + ".png"
            new_file_name = file_name.split(".png")[0] + f" {expand:04d}.png"
            if os.path.isfile(new_file_name):
                continue
    return new_file_name
    
def main():
    
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])

    seed_everything(args.seed)
    
    # Create the seeds list
    seedlist = np.append(args.seed, np.asarray(torch.randint(1, 999999999, (args.n_iter * args.n_samples - 1, ) ) ) )
    sys.stdout.write("Using seeds:\n")
    sys.stdout.write(' '.join(str(item) for item in seedlist))
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    sys.stdout.write("Setting callback ...\n")
    sys.stdout.flush()

    callback = make_callback(sampler=args.sampler,
                            dynamic_threshold=args.dynamic_threshold, 
                            static_threshold=args.static_threshold)                            

    sys.stdout.write("Loading model ...\n")
    sys.stdout.flush()

    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}")
    model.embedding_manager.load(args.embedding_path)
    #fix for using less VRAM 1/3 - add next line
    model.half()
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device:', device)
    sys.stdout.flush()
    """
    model = model.to(device)

    sys.stdout.write("Creating sampler ...\n")
    sys.stdout.flush()

    if args.sampler == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    if args.sampler in ["k_lms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
        model_wrap = K.external.CompVisDenoiser(model)
        sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.outdir

    batch_size = args.n_samples
    n_rows = args.n_rows if args.n_rows > 0 else batch_size
    if not args.from_file:
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = 0 #len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None

    if args.init_img is not None:
        assert os.path.isfile(args.init_img)
        init_image = load_img(args.init_img).to(device)
        #init_image = init_image.resize((args.W, args.H), resample=PIL.Image.LANCZOS)
        #init_image = repeat(init_image, '1 ... -> b ...', b=batch_size) #original
    
        #fix for using less VRAM 2/3 next line added
        with torch.cuda.amp.autocast(): # needed for half precision!
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

        assert 0. <= args.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(args.strength * args.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

    else:
        if args.fixed_code:
            start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)


    sys.stdout.write(f"Using {args.sampler} sampler\n")
    sys.stdout.flush()

    precision_scope = autocast if args.precision=="autocast" else nullcontext
    with torch.no_grad():
        
        #with precision_scope("cuda"):
        #fix for using less VRAM 3/3 - change previous line to this
        with torch.cuda.amp.autocast():

            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                #for n in trange(args.n_iter, desc="Sampling"):
                #    for prompts in tqdm(data, desc="data"):
                for n in range(args.n_iter):
                    torch.cuda.manual_seed(seedlist[n * args.n_samples])    #set current batch seed
                    send_seeds = seedlist[n*args.n_samples:n*args.n_samples+batch_size] #if grid, grab the relevant subset of seeds for this batch iteration
                    for prompts in data:
                        uc = None
                        if args.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        """
                        # weighted sub-prompts
                        subprompts,weights = split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            # i dont know if this is correct.. but it works
                            c = torch.zeros_like(uc)
                            # get total weight for normalizing
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(0,len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:
                             c = model.get_learned_conditioning(prompts)
                        """
                                                
                        c = model.get_learned_conditioning(prompts)
                        
                        #k_lms etc sampling (ddim and plms down below)
                        if args.sampler in ["k_lms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                            #no init
                            if args.init_img is None:
                        
                                shape = [args.C, args.H // args.f, args.W // args.f]
                                sigmas = model_wrap.get_sigmas(args.ddim_steps)

                                x = create_tensors(shape, seeds=send_seeds, step=n+1) * sigmas[0]
                                model_wrap_cfg = CFGDenoiser(model_wrap)
                                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': args.scale}
                            
                                if args.sampler=="k_lms":
                                    samples = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                                elif args.sampler=="dpm2":
                                    samples = K.sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                                elif args.sampler=="dpm2_ancestral":
                                    samples = K.sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                                elif args.sampler=="heun":
                                    samples = K.sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                                elif args.sampler=="euler":
                                    samples = K.sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                                elif args.sampler=="euler_ancestral":
                                    samples = K.sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)

                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples = accelerator.gather(x_samples)
                            #init image
                            else:
                                # encode (scaled latent)
                                z_enc = create_tensors(shape=init_latent, seeds=send_seeds, passed_sampler=sampler, passed_t_enc=t_enc, step=n+1)
                                # decode it
                                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc,)
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples = accelerator.gather(x_samples)
                        #plms or ddim sampling
                        else:
                            #no init
                            if args.init_img is None:
                                shape = [args.C, args.H // args.f, args.W // args.f]
                                start_code=create_tensors(shape, seeds=send_seeds, step=n+1)  #call the function to create the relevant tensors 
                                samples, _ = sampler.sample(S=args.ddim_steps,
                                                conditioning=c,
                                                batch_size=args.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc,
                                                eta=args.ddim_eta,
                                                x_T=start_code,
                                                seeds=send_seeds,
                                                img_callback=callback)
                                                
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            #init image
                            else:
                                # encode (scaled latent)
                                z_enc = create_tensors(shape=init_latent, seeds=send_seeds, passed_sampler=sampler, passed_t_enc=t_enc, step=n+1)
                                
                                # decode it
                                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc,)
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                
                    
                    
                    #add latest individual image to all_samples for grid creation later
                    if not args.skip_grid:
                        all_samples.append(x_samples)

                    #save next individual image
                    if args.n_samples > 0:
                        if not args.skip_save:
                            for ite, x_sample in enumerate(x_samples):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                #sys.stdout.write(f'Saving individual {args.image_file[:-4]} {args.seed+base_count}.png\n')
                                #sys.stdout.flush()
                                currseed = send_seeds[ite]
                                
                                file_name=os.path.join(sample_path, f"{args.image_file[:-4]} {currseed}.png")
                                
                                sys.stdout.write(f'Saving individual image "{get_unique_filename(file_name)}"...\n')
                                sys.stdout.flush()

                                Image.fromarray(x_sample.astype(np.uint8)).save(get_unique_filename(file_name))

                                sys.stdout.write('Saved individual image\n')
                                sys.stdout.flush()

                #create and save grid image
                if not args.skip_grid:
                    sys.stdout.flush()
                    #sys.stdout.write(f'Saving to {args.image_file}\n')

                    if args.n_samples > 1:
                        file_name=os.path.join(outpath, f"{args.image_file[:-4]} {args.seed} grid.png")
                    else:
                        file_name=os.path.join(outpath, f"{args.image_file[:-4]} {args.seed}.png")
                    
                    sys.stdout.write(f'Saving grid image "{get_unique_filename(file_name)}" ...\n')
                    sys.stdout.flush()

                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                   
                    Image.fromarray(grid.astype(np.uint8)).save(get_unique_filename(file_name))
                    
                    grid_count += 1

                    sys.stdout.flush()
                    sys.stdout.write('Saved grid image\n')
                    sys.stdout.write('Seeds reminder:\n')
                    sys.stdout.write(' - '.join(str(item) for item in seedlist))
                    sys.stdout.flush()

                toc = time.time()


if __name__ == "__main__":
    main()

#print("\n\n"+torch.cuda.memory_summary())
