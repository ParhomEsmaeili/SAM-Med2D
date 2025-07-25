from pathlib import Path
from typing import Sequence
import torch
import numpy as np
import torch.nn.functional as F
import copy 
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from argparse import Namespace
import nibabel as nib
# from loguru import logger
import gc 
import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from segment_anything import sam_model_registry as registry_sammed2d
########################################
from monai.data import MetaTensor 
import re
from itertools import product 
# from segment_anything import SamPredictor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings 
#Sanity checking:

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def sanity_check(im_slice:np.ndarray, prompts:dict, transpose=False):
    try: 
        points_npy = torch.cat(prompts['points'], dim=0).numpy()
        points_lbs = torch.cat(prompts['points_labels'], dim=0).numpy()
    except:
        pass 
    try:
        scribbles_npy = torch.cat(prompts['scribbles'], dim=0).numpy()
        scribbles_lbs = torch.cat(prompts['scribbles_labels'], dim=0).numpy() 
    except:
        pass
    try:
        bbox_npy = [i[0].numpy() for i in prompts['bboxes']]
    except:
        pass 
    if transpose:
        im_slice = im_slice.T
    im_slice = ((im_slice - im_slice.min())/(im_slice.max() - im_slice.min() + 1e-6))[...,np.newaxis]
    im_slice = np.repeat(im_slice, 3, axis=-1)
    plt.figure()
    plt.imshow(im_slice)
    try:
        show_points(points_npy, points_lbs, plt.gca())
    except:
        pass
    try:
        show_points(scribbles_npy, scribbles_lbs, plt.gca())
    except:
        pass 
    
    try:
        for box in bbox_npy:
            show_box(box, plt.gca())
    except:
        pass 
    print(f'plotted prompts according to RAS coordinates order on image slice which has transposed property {transpose} inside the plotting function.')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'RAS_prompts_on_plotting_img_transpose_{transpose}.png'))
    plt.close()


def sanity_check_output(im_slice:np.ndarray, im_type:str, prompts:dict, transpose=False):
    try: 
        points_npy = torch.cat(prompts['points'], dim=0).numpy()
        points_lbs = torch.cat(prompts['points_labels'], dim=0).numpy()
    except:
        pass
    try:
        scribbles_npy = torch.cat(prompts['scribbles'], dim=0).numpy()
        scribbles_lbs = torch.cat(prompts['scribbles_labels'], dim=0).numpy() 
    except:
        pass
    try:
        bbox_npy = [i[0].numpy() for i in prompts['bboxes']]
    except:
        pass 
    if transpose:
        im_slice = im_slice.T
    im_slice = ((im_slice - im_slice.min())/(im_slice.max() - im_slice.min() + 1e-6))[...,np.newaxis].cpu()
    im_slice = np.repeat(im_slice, 3, axis=-1)
    plt.figure()
    plt.imshow(im_slice)
    try:
        show_points(points_npy, points_lbs, plt.gca())
    except:
        pass 
    try:
        show_points(scribbles_npy, scribbles_lbs, plt.gca())
    except:
        pass    
    try:
        for box in bbox_npy:
            show_box(box, plt.gca())
    except:
        pass 
    print(f'plotted output probability map which has transposed property {transpose} inside the plotting function.')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'output_{im_type}_map_transpose_{transpose}.png'))
    plt.close()
    
def sanity_check_post_map(im_slice:torch.Tensor, prompts:dict, transpose=False):
    #We assume a slightly different structure as the image slice and prompts are assumed to be ready for injection into the network. So we need modify func above.
    try:
        points_npy = prompts['points'].numpy()
        points_lbs = prompts['points_labels'].numpy()
    except:
        pass 
    try:
        scribbles_npy = prompts['scribbles'].numpy()
        scribbles_lbs = prompts['scribbles_labels'].numpy()
    except:
        pass
    try:
        #bbox is not in a list now, it comes in a N x 2N_dim shape torch tensor.
        bbox_npy = [i.numpy() for i in torch.unbind(prompts['bboxes'], axis=0)]
    except:
        pass 
    
    im_slice_npy = np.moveaxis(im_slice[0].numpy(), 0, -1)
    if transpose:
        im_slice_npy = np.swapaxes(im_slice_npy, 0, 1) #Standard transposition will actually send the rgb channels back into axis=0, so lets not do this! :)
    im_slice_npy = ((im_slice_npy - im_slice_npy.min())/(im_slice_npy.max() - im_slice_npy.min() + 1e-6))
    plt.figure()
    plt.imshow(im_slice_npy)
    
    try:
        show_points(points_npy, points_lbs, plt.gca())
    except:
        pass 
    try:
        show_points(scribbles_npy, scribbles_lbs, plt.gca())
    except:
        pass
    try:
        for box in bbox_npy:
            show_box(box, plt.gca())
    except:
        pass 
    print(f'plotted prompts according to mapped (resized) RAS coordinates order on resized image slice which has transposed property {transpose} inside the plotting function.')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'model_dom_prompts_on_plotting_img_transpose_{transpose}.png'))
    plt.close()

     

def load_sammed2d(checkpoint_path, encoder_model_type, image_size, device="cuda"):
    args = Namespace()
    args.image_size = image_size
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    model = registry_sammed2d[encoder_model_type](args).to(device)
    model.eval()

    return model


class InferApp:
 
    def __init__(self, dataset_info, infer_device):
        
        self.sanity_check = False #True #False #True
        self.sanity_slice_check = 510#522 #3 #39
        self.cpu_gpu_burden_ratio = 1 #1 = put everything on GPU with the modulo operation! 1 is a prime factor of every natural number.
        
        #Some hardcoded params only for performing a sanity check on mapping the input image domain to the domain expected by the model.

        ############ Initialising the inference application #####################
        self.dataset_info = dataset_info
        self.infer_device = infer_device

        if self.infer_device.type != "cuda":
            raise RuntimeError("segvol can only be run on cuda.")

        #Setting image configurations which will be used for configuring the sam model. 
        
        #Setting the names for the corresponding indices of the input image arrays (we will assume that the inputs will be oriented in RAS convention)
        
        index_to_plane = {
            0:'sagittal',
            1:'coronal',
            2:'axial'
        }

        image_size = 256
        model_dom_size = (image_size, image_size)
        image_axes = (2,) 
        
        if any([i!=2 for i in image_axes]):
            warnings.warn('Image axes != 2 selected, but the operations performed to map the coordinates into the model domain are extrapolated from ax=2, major warning.')
        if len(image_axes) > 1:
            raise Exception('There is no existing strategy for performing fusion of segmentations across different planes, only a singular plane can be processed.')
        
        self.app_params = {
            'encoder_model_type': "vit_b",
            'checkpoint_path': "sam-med2d_b.pth",
            'image_size': model_dom_size,
            'image_axes': {k:index_to_plane[k] for k in image_axes}
        } 
        #Loading inference model.
        self.model = self.load_model()
        self.build_inference_apps()

        ########################################################################## 

        #Some assumptions we make are listed below when they have an actionable behaviour. Some however, are not: E.g., one assumption we make is that while we will
        #enforce that the bbox will remain static post initialisation of a slice and for which we remove any repeats since it would constitute the generation of extra
        #instances, we do not do this for the points/scribbles. There may be instances where a user is with insistence trying to repeatedly click on the same point!
        #
        # Implicitly we are using the original assumption of the model, which is that bboxes are strictly for constraining, and not for editing!
    

        #Initialising any remaining variables required for performing inference.

        self.autoseg_infer = False #This is a variable for storing the action taken in the instance where there is no prompting information provided in a slice.
        #In the case where it is True, a prediction will be made, and the stored pred and output pred will be the same.
        #In the case where it is False, a prediction will not be made, the stored pred will be None, and the output pred will be zeroes.
        
        self.static_bbox = True #This is a variable denoting whether we will be permitting for the use of bboxes which are dynamic throughout refinement process. 
        #NOTE: This is within a given slice, it is entirely plausible that the set of bboxes would change throughout if annotation was occuring on a slice-by-slice basis.

        #In the case where it is False, it would indicate that the provided bboxes can be dynamic post slice-level initialisation (at the slice level, clearly for a 
        # volume a slice-by-slice method could be dynamic in the sense that the slice being segmented on could change!) between iterations.
        #In the case where it is True, it must be static (at the slice level) between iterations. I.e., if it is not the first set of bboxes in that slice it should 
        # raise an exception. 

        self.prop_freeform_uniformly = True #This is a variable denoting whether the free-form prompts, like clicks and scribbles are uniformly distributed across
        #the set of instance-level closed contour prompts (i.e. bbox) under the assumption that we are working under semantic segmentation constraints rather than instance segmentation.
        # I.e., that a single class could have multiple instances, and as such the free-form prompts are not as strictly separated as a bbox. Unused parameter, only
        #provided to store the value during validation. Always defaults to True.

        self.split_forward_mask = True #This is a variable denoting whether the lowres mask that is forward propagated (in cases where we are actually doing this) will
        #be split by bbox quantity. 
        # Variable was initially introduced as it was not possible to modify the quantity of mask channels, but a modification to prediction script was made such that in 
        # cases where a slice was stored with a single channel, that the introduction of bboxes would introduce the assumption that we could distribute this as the 
        # initialisation mask and then carry forward the corresponding mask. It would however require that the quantity of boxes remain fixed after a box is 
        # introduced to a given slice.

        self.multimask_output_always = True  # By default, it will be true to match defaults from demo, this means that masks are always generated by propagating 
        #through each of the MLPs (3) and then chosen according to the best predicted iou. 

        if self.split_forward_mask:
            if self.autoseg_infer:
                warnings.warn('Highly experimental, if autoseg_infer=True, lowres masks are actually stored internally (rather than None), since this was previously not compatible with a strategy which passes through individual lowres masks for each bbox (as it would require a mask for each) proceed with caution.')

        #Some preprocessing, post processing params.
        self.clip_lower_bound = 0.5
        self.clip_upper_bound = 99.5

        #some default ct clamp values..:
        # window_width = 400
        # window_level = 40 
        # lower_bound = window_level - window_width / 2
        # upper_bound = window_level + window_width / 2
        # self.default_ct_clamp = (lower_bound, upper_bound)
        
        # self.image_embeddings_dict = {}
        self.permitted_prompts = ('points', 'bboxes', 'scribbles')
        self.glob_norm_bool = True
        self.slice_norm_method = None 

        self.pixel_mean, self.pixel_std = (
            self.model.pixel_mean.squeeze().cpu().numpy(),
            self.model.pixel_std.squeeze().cpu().numpy(),
        )

        self.mask_threshold_sigmoid = 0.5 

        self.app_params.update({
            'autoseg_infer_bool':self.autoseg_infer,
            'static_bbox':self.static_bbox,
            'prop_freeform_prompts_uniformly': self.prop_freeform_uniformly,
            'split_forward_mask':self.split_forward_mask,
            'multi_ambig_mask_always':self.multimask_output_always,
            'permitted_prompts':self.permitted_prompts,
            'intensity_lower_quantile': self.clip_lower_bound,
            'intensity_upper_quantile': self.clip_upper_bound,
            'pixel_normalisations':{
                'mean':self.pixel_mean,
                'std': self.pixel_std
            },
            # 'ct_default_clamp': self.default_ct_clamp,
            'glob_norm_bool': self.glob_norm_bool,
            'slice_norm_method': self.slice_norm_method,
            'prob_thresh': self.mask_threshold_sigmoid, 
            'sanity_check_slice': self.sanity_slice_check
        })
        
    def app_configs(self):
        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return self.app_params 
    
    def load_model(self):
        #Just in case of any spooky action at a distance since we have not yet containerised this application.
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        return load_sammed2d(checkpoint_path=os.path.join(base_dir, 'ckpt', self.app_params['checkpoint_path']), 
                                   encoder_model_type=self.app_params['encoder_model_type'], 
                                   image_size=self.app_params['image_size'][0],
                                   device=self.infer_device)

    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_predict':self.binary_inference},
            'IS_interactive_init': {'binary_predict':self.binary_inference},
            'IS_interactive_edit': {'binary_predict':self.binary_inference}
            }
    
    
    def binary_inference(self, request):
        #Mapping the input request to the model domain:
        init_bool, infer_slices, affine = self.binary_subject_prep(request=request)
        #The input information has been stored separately since we will iteratively update this throughout the refinement process.

        #Extracting prediction slices:
        self.binary_predict(init_bool, infer_slices)

        #Converting the set of prediction and probability map slices into the output volume:
        discrete_mask, prob_mask = self.binary_merge_slices()
        
        return discrete_mask, prob_mask, affine 
    def binary_subject_prep(self, request:dict):
        
        #Here we perform some actions for determining the state of the infer call for adjusting some of our info extraction mechanisms.
         
        #Ordering the set of interaction states provided, first check if there is an initialisation: if so, place that first. 
        im_order = [] 
        init_modes  = {'Automatic Init', 'Interactive Init'}
        edit_names_list = list(set(request['im']).difference(init_modes))

        #Sorting this list.
        edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))

        #Extending the ordered list. 
        
        im_order.extend(edit_names_list) 
        #Loading the image and prompts in the input-im domain & the zoom-out domain.
        
        if request['model'] == 'IS_interactive_edit':
            key = edit_names_list[-1]
            is_state = request['im'][key]
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = False 
            
            assert isinstance(self.image_embeddings_dict, dict) and self.image_embeddings_dict
            assert isinstance(self.internal_lowres_mask_storage, dict) and self.internal_lowres_mask_storage
            assert isinstance(self.internal_discrete_output_mask_storage, dict) and self.internal_discrete_output_mask_storage
            assert isinstance(self.internal_prob_output_mask_storage, dict) and self.internal_prob_output_mask_storage
            assert isinstance(self.orig_prompts_storage_dict, dict) and self.orig_prompts_storage_dict
            assert isinstance(self.model_prompts_storage_dict, dict) and self.model_prompts_storage_dict #Just asserting that these are dicts and also non-empty.
            assert isinstance(self.box_prompted_slices, dict) and self.box_prompted_slices 

        elif request['model'] == 'IS_interactive_init':
            key = 'Interactive Init' 
            is_state = request['im'][key]
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = True 
            
            try:
                del self.image_embeddings_dict
                del self.internal_lowres_mask_storage
                del self.internal_discrete_output_mask_storage
                del self.internal_prob_output_mask_storage
                del self.orig_prompts_storage_dict
                del self.model_prompts_storage_dict
                
                torch.cuda.empty_cache() #We clear cache for each new case because image sizes can have variance! 
            except:
                pass #HACK: Not a good solution but want to clear the cached memory while keeping the script "online".

            self.image_embeddings_dict = dict()
            self.internal_lowres_mask_storage = dict() #This is in the model domain!
            self.internal_discrete_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            self.internal_prob_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            
            self.box_prompted_slices = dict() #Stores the set of slices which have been prompted already, this is relevant for the bbox placement as it will be assumed
            #that in cases where it is static, that incoming bbox prompts will be cross-examined accordingly.
            self.orig_prompts_storage_dict = dict()
            self.model_prompts_storage_dict = dict()

            torch.cuda.empty_cache()

        elif request['model'] == 'IS_autoseg':
            key = 'Automatic Init'
            is_state = request['im'][key]
            if is_state is not None:
                raise Exception('Autoseg should not have any interaction info.') 
            
            #actually just raise an exception, it can't handle autoseg.
            raise Exception('True Autoseg (not the S.A.T) is too OOD for this algorithm')
            init = True 

            del self.image_embeddings_dict
            del self.internal_lowres_mask_storage
            del self.internal_discrete_output_mask_storage
            del self.internal_prob_output_mask_storage
            del self.orig_prompts_storage_dict
            del self.model_prompts_storage_dict
            
            torch.cuda.empty_cache() #We clear cache for each new case because image sizes can have variance! 


            self.image_embeddings_dict = dict()
            self.internal_lowres_mask_storage = dict() #This is in the model domain!
            self.internal_discrete_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            self.internal_prob_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            
            self.box_prompted_slices = dict() #Stores the set of slices which have been box-prompted already, this is relevant for the bbox placement as it will be assumed
            #that in cases where it is static, that incoming bbox prompts will be cross-examined accordingly.
            self.orig_prompts_storage_dict = dict()
            self.model_prompts_storage_dict = dict() 
        #Mapping image and prompts to the model's coordinate space. NOTE: In order to disentangle the validation framework from inference apps 
        # this is always assumed to be handled within the inference app.

        infer_slices = self.binary_prop_to_model(request['image'], is_state, init)        
        #The images, image embeddings, and prompt information are all stored in attributes separately. 
        for ax in self.app_params['image_axes']:
            print(f'altered slices for ax {ax} are {infer_slices[ax]}')

        affine = request['image']['meta_dict']['affine']
        return init, infer_slices, affine  

    
    def binary_prop_to_model(self, im_dict: dict, is_state: dict | None, init: bool):
        
        #Prompts and images are provided in L->R, P->A, S -> I  (where the image itself was also been correspondingly rotated since array_coords >=0).
        # 
        #Visual inspection of the images provided in the demo demonstrate that the positive directions of the axes in the axial slice corresponds to the R -> L, A -> P convention.
        #but, the ordering of the axes differs. I.e., the y dimension of the image array is the A - P dimension, while the x dimension is the R -> L dimension.

        #Note that OpenCV convention is -----> x, therefore an array that is M x N represents N in the X direction, and M in the y direction.
                                    #  |
                                    #  |
                                    #  |
                                    #  v
                                    # y

        #We first propagate the image into the model domain, and store the image embeddings.

        if init:
            input_dom_img = im_dict['metatensor']
            # input_dom_affine = im_dict['meta_dict']['affine']
            # input_dom_shape = input_dom_img.shape[1:] #Assuming a channel-first image is being provided.
            im_slices_model_dom, input_dom_shapes = self.binary_im_to_model_dom(input_dom_img) 
            #Storing the shape of the slice in each corresponding axis being used.
            self.orig_im_shape = input_dom_img.shape[1:]
            self.input_dom_shapes = input_dom_shapes
            #Storing the image slice in the input model domain in memory for sanity checks 
            if self.sanity_check:
                self.im_slices_post_map = im_slices_model_dom 
            #Now we will extracted the image embeddings correspondingly, and store them in memory.
            self.binary_extract_im_embeddings(im_slices_model_dom=im_slices_model_dom)

        #Now we propagate the prompt information into the model domain. 

        if bool(is_state):
            p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
            
            if p_dict[0]['bboxes'] is not None and p_dict[1]['bboxes_labels'] is not None:
                #We will flag any background bboxes here. SAM-MED2D cannot handle these (nor does it have any meaning in the context within which they use this.)
                if not all([i == 1 for i in p_dict[1]['bboxes_labels']]): 
                    raise Exception('Was presented with bboxes delineating background, SAM-Med2D cannot handle this formulation of prompts.')
                # else:
                #     #There is a background bboxes, SAMMed2D doesn't understand what this means and could break the system.
                #     # bbox_list = []
                #     # bbox_lb_list = []
                #     # for i,j in zip(p_dict[0]['bboxes'], p_dict[1]['bboxes_labels']):
                #     #     if j == 1:
                #     #         bbox_list += [i] 
                #     #         bbox_lb_list += [j] 
                #     # p_dict[0]['bboxes'] = bbox_list
                #     # p_dict[1]['bboxes_labels'] = bbox_lb_list
                #     # assert p_dict[0]['bboxes'] != [] and p_dict[1]['bboxes_labels'] != []
                #     pass
            #Determine the prompt type from the input prompt dictionaries: Not sure if intersection is optimal for catching exceptions here.
            provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
            
            #
            if any([p not in self.permitted_prompts for p in provided_ptypes]):
                raise Exception(f'Non-permitted prompt was supplied, only the following prompts are permitted {self.permitted_prompts}')
        else:
            #Handling empty prompt dict and/or Autosegmentation.
            provided_ptypes = None
            p_dict = None 
            # self.binary_extract_prompts(current_prompts=None, init_bool=init)
        
        infer_slices = self.binary_extract_prompts(current_prompts=p_dict, provided_ptypes=provided_ptypes, init_bool=init)

        if self.sanity_check:
            for ax in self.app_params['image_axes']:
                sanity_check(self.im_slices_input_dom[ax][self.sanity_slice_check], self.orig_prompts_storage_dict[ax][self.sanity_slice_check], transpose=False)
                sanity_check_post_map(self.im_slices_post_map[ax][self.sanity_slice_check], self.model_prompts_storage_dict[ax][self.sanity_slice_check], transpose=False)

        return infer_slices
        # return {
        #     'im_slices_model_dom': im_slices_model_dom,
        #     'prompt_model_dom': prompt_fg_dom, 
        #     'input_dom_affine': input_dom_affine,
        #     'input_dom_shapes': input_dom_shapes, #This is the dictionary, for each axis which we extract slices for, that contains the corresponding size of the image slices.
        # }
    
    def binary_extract_prompts(self, current_prompts:dict | None, provided_ptypes:list | None, init_bool:bool):
        #Takes the current prompts, which will need to be added to the set of stored prompts.
        #Prompts and images are provided in R ->L, A->P, S -> I  (where the image itself was also been correspondingly rotated since array_coords >=0).
        

        #Visual inspection of the images provided in the demo demonstrate that the positive directions of the axes in the axial slice corresponds to the R -> L, A -> P convention.
        #but, the ordering of the axes differs. I.e., the y dimension of the image array is the A -> P dimension, while the x dimension is the R -> L dimension.

        #Note that OpenCV convention is -----> x, therefore an array that is M x N represents N in the X direction, and M in the y direction, i.e. y, x ordering.
                                    #  |
                                    #  |
                                    #  |
                                    #  v
                                    # y

        #This therefore requires transposition of the axes for the images when mapping to the model domain. We assume this is consistent
        #across the axes chosen for slicing, but we will only test with the axial slices just to be careful.

        #For the prompt coordinates, they are provided x,y ordering. This corresponds to the original RAS order, so we do nothing.

        #NOTE: However, since we will have to perform a rescaling for insertion to the SAMMed2D model domain, we will need to perform a rescaling with the scale factors
        #being computed on the transposed image slice coordinates.

        #First we stash the set of prompts in our collected bank in the input image domain.
        infer_slices = self.binary_store_prompts(current_prompts=current_prompts, provided_ptypes=provided_ptypes, init_bool=init_bool)
        if init_bool:
            for ax in self.app_params['image_axes']:
                if self.orig_im_shape[ax] != len(infer_slices[ax]):
                    raise Exception(f'The quantity of altered slices in the initialisation for axis {ax} was {len(infer_slices[ax])}, but it needs to be {self.orig_im_shape[ax]}')
        
        #Now we need to map this into the domain of the expected images, which are defined by the array definition of cv2, and then map that to the model domain.
        self.binary_map_prompt_to_model(infer_slices=infer_slices, init_bool=init_bool)
        return infer_slices
    
    def binary_store_prompts(self, current_prompts: dict | None, provided_ptypes:list | None, init_bool:bool):
        #This function stores the prompts in the original image domain, but according to the corresponding slicing axes.
        if init_bool:
            infer_slices = dict()

            for ax in self.app_params['image_axes']:
                infer_slices[ax] = list(range(len(self.image_embeddings_dict[ax]))) #All slices are going to be used for inference for initialisation!
                self.box_prompted_slices[ax] = {k:False for k in range(len(self.image_embeddings_dict[ax]))} #We start with the assumption that a slice is not bbox prompted,
                #and update this corresponding to the observation. This is required for cases where the assumed structure of a bbox interaction is that of a "grounding"
                #spatial prior.
                
                #Init the dict for the given axis always
                current_ax_dict = dict()
                for slice_idx in range(len(self.image_embeddings_dict[ax])):
                    #In this case we are going to initialise an empty set of prompts for each axis and slice, across all prompt types. 
                    current_ax_dict[slice_idx] = {k:[] for k in list(self.permitted_prompts) + [f'{i}_labels' for i in self.permitted_prompts]}
                self.orig_prompts_storage_dict[ax] = current_ax_dict
                #Initialising the storage dict for the prompts!
                
        if current_prompts is not None: #We could have used provided ptypes list = None also, this denotes the autoseg case!
            if not init_bool:
                infer_slices = {ax:[] for ax in self.app_params['image_axes']} 
                #We create a dict to store the set of slices which were modified for editing so that we are not performing inference on slices which were not prompted.
            
            for ax in self.app_params['image_axes']:  
                #If prompts are not none, we require special treatment according to the prompt type as bbox might be in 3D and needs to be collapsed to a set of 2D!
                
                box_prompted_slices = [] 
                #Initialisation of an empty list containing the list of slice indices for the given axis which contain new initialisation bbox, this is required 
                # such that after we pass through the set of inputted bbox prompts, we can update our memory bank of the slices which have been bbox initialised. 
                # This is also critical for checking that no new bboxes are being placed if we configure the bboxes as being slice-level static!
                    
                for ptype in provided_ptypes:
                    
                    # print(ptype)
                    if ptype == 'points' or ptype == 'scribbles':
                        #We treat scribbles in a very similar capacity as points, but for now we will keep these as separate items in memory.

                        if current_prompts[0][ptype] is None or current_prompts[1][f'{ptype}_labels'] is None:
                            raise Exception('Uncaught instance of no prompts being available for extraction, should have been flagged or handled earlier!')
                        #Extract the list of prompt items, concatenate together for vectorising indexing.
                        p_cat = torch.cat(current_prompts[0][ptype], dim=0)
                        if ptype == 'points':
                            p_lab_cat = torch.cat(current_prompts[1][f'{ptype}_labels'])
                        else:
                            #In the downstream label extraction code, since we will convert a scribble to a set of points, we must also convert the corresponding labels
                            #too.
                            #First extract the length of each scribble in terms of #N_points, then use repeat interleave to expand the set of labels for a 1-to-1 map of points and labels
                            p_lab_cat = torch.repeat_interleave(torch.tensor(current_prompts[1][f'{ptype}_labels']), torch.tensor([p.shape[0] for p in current_prompts[0][ptype]]))
                        
                        #Denoting the required axes when extracting the prompt locations within a given slice.
                        required_ax = list(set([0,1,2]) - set([ax])) 
                        #Required coord component corresponds to the set difference between [0,1,2] and [ax] (where ax is the axial dimension for which we are extracting slices.)
                        
                        #Extracting the set of valid axis slice coordinates to reduce our search:
                        valid_slices = p_cat[:, ax].unique() 
                        for slice_idx in valid_slices.tolist():
                            #Extract the set of coords which have axis dimension equivalent to the slice index.
                            valid_p_idxs = torch.argwhere(p_cat[:,ax] == slice_idx)
                            
                            if valid_p_idxs.numel():
                                #Setting the slices which require inference.
                                if slice_idx in infer_slices[ax]:
                                    #The logic here, is just to append the slice if the slice was not already being tracked, according to whether a prompt exists within
                                    #the slice. It will NOT skip the insertion of the prompt! Hence why we have used a pass explicitly, rather than a continue!
                                    pass
                                else:
                                    infer_slices[ax] += [slice_idx]

                                if valid_p_idxs.numel() > 1:
                                    ps = [p.unsqueeze(0) for p in p_cat[tuple(valid_p_idxs.T)][:, required_ax].unbind(dim=0)]
                                    self.orig_prompts_storage_dict[ax][slice_idx][ptype] += ps
                                    self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] += [p.unsqueeze(0) for p in p_lab_cat[tuple(valid_p_idxs.T)].unbind(dim=0)]
                                    #These lines are extracting the valid set of point coordinates (or labels) according to the condition, extracting the relevant 
                                    # coordinates or labels for the slice and then unrolling into a list of tensors with shape [1,2] or shape [1] respectively.

                                    if not all([(i[0].numpy() <= self.input_dom_shapes[ax]).all() for i in ps]):
                                        Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                                elif valid_p_idxs.numel() == 1:
                                    ps = [p_cat[tuple(valid_p_idxs.T)][:, required_ax]]
                                    self.orig_prompts_storage_dict[ax][slice_idx][ptype] += ps
                                    self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] += [p_lab_cat[tuple(valid_p_idxs.T)]]
                                    #If using scribbles, it can be the exact same, as a scribble represented as a single point is equivalent to treatment as such!
                                    #, hence it will only correspond to a singular label.
                                    if not all([(i[0].numpy() <= self.input_dom_shapes[ax]).all() for i in ps]):
                                        Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                    elif ptype == 'bboxes':
                        #Denoting the required axes for extracting the prompt locations within a given slice.
                        required_ax = list(set([0,1,2,3,4,5]) - set([ax, ax + 3])) 
                        #Required coord component corresponds to the set difference between [0,1,2,3,4,5] and [ax, ax+3] 
                        # (where ax is the axial dimension for which we are extracting slices, this is because the bbox is provided in min_r, min_a, min_s, max_r,max_a,max_s convention)
                        
                        #Slightly different treatment to the points and scribbles, we loop over the set of bboxes instead as we cannot disentangle the structure of a bbox
                        # for inference like we did for the scribbles..... 
                        for bbox, bbox_label in zip(current_prompts[0][ptype], current_prompts[1][f'{ptype}_labels']):
                            box_extrema = bbox[0]
                            #Catching any cases where a 2D bbox is passed through in a 3D format.
                            extrema_matching = [box_extrema[i] == box_extrema[i+3] for i in range(3)]
                            #If the extrema are matching in the axis we are currently using, great! Just take the bbox of that slice. Otherwise, we continue because
                            #a 2D bounding box is just a line when being sliced along axes orthogonal to the plane of the bbox. 
                            if sum(extrema_matching) > 1:
                                #In this scenario, somehow we have been provided with a 3D bbox that is actually just a line... 
                                # #NOTE: the choice could be made to convert this to a scribble or point but we will not.
                                warnings.warn('Somehow have been provided with a bounding box set of extrema which describes a line or point, please check.')
                                continue
                            elif sum(extrema_matching) == 1:
                                if list(filter(lambda i: extrema_matching[i], range(len(extrema_matching))))[0] != ax:
                                    #In this case, the axis where the extrema were matching was NOT the same axis we are currently inspecting..., so continue,
                                    # otherwise we are just getting a set of lines..... 
                                    # NOTE: the choice could be made to convert this to a scribble but we will not. 
                                    continue 
                                    #We use continue here to skip over this bbox completely! Not a pass.
                                else:
                                    #In this case, the axis was matching, so we can potentially insert at that slice the corresponding bbox according to extrema on the 
                                    # other axes

                                    #We set the slice_idx for this bbox to be given by the value of the "extrema" for the given axis.
                                    slice_idx = int(box_extrema[ax]) 
                                    #We extract the bbox parameterisation.
                                    bbox = box_extrema[[i for i in range(box_extrema.shape[0]) if i not in [ax, ax + 3]]]
                                    #Now we check for any faults and add in any bboxes appropriately. 
                                    if self.static_bbox:
                                        #First we perform a check to see if the corresponding slice has had a set of bboxes it was previously prompted with already, 
                                        # if not then append it to the list of slices. NOTE: Since it is possible for multiple bboxes to be used, the update occurs 
                                        # after all bboxes are processed.
                                        if self.box_prompted_slices[ax][slice_idx]:
                                            #In this case we must perform a check to make sure that the bbox being provided is not different from that which already
                                            #was placed. If it is not distinct then just continue.
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. 
                                                # SAMMed2D is incompatible with this and will just keep growing the quantity of prediction masks.
                                                #It is unclear what the procedure should be with a changing quantity of bboxes, as forward pass creates a masks for each bbox. 
                                            else:
                                                #If the bbox was not, and the slice has already been box-prompted, since we are in the static_bbox config we raise an
                                                #exception.
                                                raise Exception(f'A new bounding box was provided in axis {ax}, slice {slice_idx} in an iteration after the initial instance where the slice had been bbox-prompted.')
                                        else:
                                            #In this case then it is perfectly acceptable to just append the bbox prompts because the given slice has not been previously bbox initialised!
                                            #We still check if the bbox hasn't already been placed somehow anyways to prevent repeats!
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. 
                                                # SAMMed2D is incompatible with this and will just keep growing the quantity of prediction masks.
                                                #It is unclear what the procedure should be with a changing quantity of bboxes, as forward pass creates a masks for each bbox. 
                                            else:
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes'] += [bbox.unsqueeze(0)] #We will use a 1 x N_dim convention
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes_labels'] += [bbox_label]

                                            #Appending the slice to the set of slices which are bbox initialised, we will later consider only the unique indices so any
                                            #redundancy is not important. We do not update the bool here as we do not want to throw an exception as we loop through bboxes.
                                            box_prompted_slices += [slice_idx] 
                                    else:
                                        raise NotImplementedError('The handling of non-static bboxes (post-initialisation) is not supported within a given slice.')       
                                
                                    if slice_idx in infer_slices[ax]:
                                        #The logic here, is just to append the slice if the slice was not already being tracked, according to whether a new prompt 
                                        # exists within the slice. It will NOT skip the insertion of the prompt! Hence why we have used a pass explicitly, 
                                        # rather than a continue!
                                        pass 
                                    else:
                                        infer_slices[ax] += [slice_idx]
                                    if not all([bbox[i] <= bound and bbox[i + 2] <= bound for i, bound in enumerate(self.input_dom_shapes[ax])]):
                                        raise Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                            else:
                                #In this case, no extrema were matching, we can proceed as standard:

                                for slice_idx in range(box_extrema[ax],box_extrema[ax + 3] + 1):    
                                #Since the bounding box is consistent in shape across the slices along a given axis, we can just insert as expected across all corresponding slices bounded
                                #by the extrema of the bounding box along the given axis.
                                    bbox = box_extrema[[i for i in range(box_extrema.shape[0]) if i not in [ax, ax + 3]]]
                                    
                                    #Now we check for any faults, or add any bboxes by unroll a 3d bbox into 2d slices and checking each slice to see if it has been
                                    #bbox initialised, and whether it has any dynamically generated bboxes within the slice.
                                    if self.static_bbox:
                                        if self.box_prompted_slices[ax][slice_idx]:
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. SAMMed2D is incompatible with this,
                                                #and it is unclear what the procedure should be with variable quantities of bboxes as forward pass creates N masks for each bbox. 
                                            else:
                                                raise Exception(f'A new bounding box was provided in axis {ax}, slice {slice_idx} in an iteration after the initial instance where the slice had been bbox-prompted.')
                                        else:
                                            #In this case then it is perfectly acceptable to just append the bbox prompts because the given slice has not been 
                                            # previously bbox initialised! We still check if the bbox hasn't already been placed somehow anyways to prevent repeats!
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. 
                                                # SAMMed2D is incompatible with this and will just keep growing the quantity of prediction masks.
                                                #It is unclear what the procedure should be with a changing quantity of bboxes, as forward pass creates a masks for each bbox.
                                            else:
                                                #If it was not bbox initialised in a prior iteration of inference (or if there wasn't one already) then we can freely add
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes'] += [bbox.unsqueeze(0)] 
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes_labels'] += [bbox_label]

                                            #Appending the slice to the set of slices which are bbox initialised, we will later consider only the unique indices so any
                                            #redundancy is not important. We do not update the bool here as we do not want to throw an exception as we loop through bboxes.
                                            box_prompted_slices += [slice_idx] 
                                    else:
                                        raise NotImplementedError('The handling of non-static bboxes (post-initialisation) is not supported within a given slice.') 
                                    if slice_idx in infer_slices[ax]:
                                        #The logic here, is just to append the slice if the slice was not already being tracked, according to whether a prompt exists within
                                        #the slice. It will NOT skip the insertion of the prompt! Hence why we have used a pass explicitly, rather than a continue!
                                        pass 
                                    else:
                                        infer_slices[ax] += [slice_idx]
                                    
                                    if not all([bbox[i] <= bound and bbox[i + 2] <= bound for i, bound in enumerate(self.input_dom_shapes[ax])]):
                                        raise Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                    else:
                        raise Exception('Somehow, a non-supported prompt type managed to get through.')
        
                #Here we commit the stashed set of slices which have been bbox initialised.
                for idx in set(box_prompted_slices): self.box_prompted_slices[ax][idx] = True  
        
        
        for i in self.app_params['image_axes']: infer_slices[i].sort() 
        return infer_slices
    
    def binary_map_prompt_to_model(self, infer_slices: dict, init_bool: bool):
        #This function maps the coordinates in the image domain, to the coordinates in the model domain according to the image size configuration selected initially.

        #The second value in the shape of the img array in the cv2 domain corresponds to the first value in the RAS ordering (e.g. in RA plane, R,   in AS plane, A, 
        # in RS plane, R). The first value in the shape corresponds to the second value in the RAS ordering (e.g. in RA plane, A,  in AS plane, S   in RS plane, R)
        
        #This is reflected in the methods used for performing these transformations called.

        for ax in self.app_params['image_axes']:
            #For each axis, we store the set of prompts in the altered slices.
            if init_bool:
                self.model_prompts_storage_dict = copy.deepcopy(self.orig_prompts_storage_dict)
                #We copy as this will copy over the structure, but this does not mean that our job is yet complete, for autoseg it will though.
            for slice_idx in infer_slices[ax]:
                #This isn't efficient, but we are just going to recompute the mapped values every single time for the set of altered slices 
                # otherwise it would require that we needed to have outline which were the new prompts. 
                if all([i == [] for i in self.orig_prompts_storage_dict[ax][slice_idx].values()]):
                    pass #In this case, just pass over, we copied over the empty list already for the autoseg modes. 
                else:
                    #In this case, there are some prompts which require mapping.
                    for ptype in self.permitted_prompts:
                        if self.orig_prompts_storage_dict[ax][slice_idx][ptype] == []:
                            continue #In this case the given prompt did not contain anything.
                        if ptype == 'points' or ptype == 'scribbles':
                            p, p_lab = self.orig_prompts_storage_dict[ax][slice_idx][ptype], self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels']
                            if len(p) > 1:
                                p_inp = torch.cat(p, dim=0)
                                p_lab_inp = torch.cat(p_lab, dim=0)
                            elif len(p) == 1:
                                p_inp = p[0]
                                p_lab_inp = p_lab[0] 
                            else:
                                raise Exception('Cannot be in the subloop for processing prompts that do not exist.')
                            #NOTE: Since we have transposed the slice extracted from the input image domain for the model domain input image, 
                            # when mapping to the model domain the corresponding coord will still have to undergo a rescaling corresponding to the scale factors 
                            # computed using the transposed image shape.
                            
                            mapped_coords = self.apply_coords(p_inp, tuple(self.input_dom_shapes[ax][::-1]), self.app_params['image_size'])
                            self.model_prompts_storage_dict[ax][slice_idx][ptype] = mapped_coords.to(dtype=torch.float) 
                            self.model_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] = p_lab_inp.to(dtype=torch.int)
                            assert torch.all(abs(mapped_coords[:,0]) <= self.app_params['image_size'][0]) and torch.all(abs(mapped_coords[:,1]) <= self.app_params['image_size'][1])
                        elif ptype == 'bboxes':
                            #For bbox we concat as in the demo, we will for now pass through the labels too even though they won't be used for passing into the prompt encoder.
                            p, p_lab = self.orig_prompts_storage_dict[ax][slice_idx][ptype], self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels']  
                            if len(p) > 1:
                                p_inp = torch.cat(p, dim=0)
                                p_lab_inp = torch.cat(p_lab, dim=0)
                                mapped_coords = self.apply_boxes(p_inp, tuple(self.input_dom_shapes[ax][::-1]), self.app_params['image_size'])
                                self.model_prompts_storage_dict[ax][slice_idx][ptype] = mapped_coords.to(dtype=torch.float) 
                                self.model_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] = p_lab_inp.to(dtype=torch.int)
                            elif len(p) == 1:
                                p_inp = p[0]
                                p_lab_inp = p_lab[0] 
                                mapped_coords = self.apply_boxes(p_inp, tuple(self.input_dom_shapes[ax][::-1]), self.app_params['image_size'])
                                self.model_prompts_storage_dict[ax][slice_idx][ptype] = torch.as_tensor(mapped_coords, dtype=torch.float)
                                self.model_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] = torch.as_tensor(p_lab_inp, dtype=torch.int) 
                            else:
                                raise Exception('Cannot be in the subloop for processing prompts that do not exist.')
                            assert torch.all(abs(mapped_coords[:,[0,2]]) <= self.app_params['image_size'][0]) and torch.all(abs(mapped_coords[:,[1,3]]) <= self.app_params['image_size'][1])
    
    def binary_im_to_model_dom(self, input_dom_im): #Mostly borrowed from RadioActive SAM-Med2D inferer.
        #Assuming that input image is in RAS convention, the axes denote the subset from (0,1,2) which denotes the axes along which slices will be taken, i.e., 
        #if (2), then the values being extracted are from the first 2 according to the third index (i.e. in the R-L/A-P plane, aka axial slices)
        
        #First removing the channel dimension and converting to a numpy array.
        input_dom_im_backend = copy.deepcopy(input_dom_im).data.numpy()[0,:]

        if self.sanity_check:
            self.im_slices_input_dom = {} #This is a variable that is not used for anything other than sanity_checking that the array channels are in the correct order. 
        slices_processed = {}
        orig_im_dims = {}

        if len(self.app_params['image_axes']) > 1:
            raise Exception('Implementation currently is not capable of simultaneous handling of > 1 planar segmentations.')
        for ax in self.app_params['image_axes']:
            if self.sanity_check:
                ax_slices_pre_resizing = {}
            ax_slices_process = {} 

            if len(self.dataset_info['task_channel']) > 1:
                raise Exception('Implementation currently is not capable of simultaneous handling of > 1 task channel segmentations.')
            #Normalisation logic is taken from their paper describing their dataset:
            if self.glob_norm_bool:
                #They just min-max norm and round up. lets add a clipping to prevent outliers from ruining it too.

                #If it is not CT (i.e. values won't be negative), then we can use the positive voxels to calculate intensity
                #statistics.
                if self.dataset_info['task_channel'][0] == 'CT':
                    lower_bound, upper_bound = np.percentile(input_dom_im_backend, self.clip_lower_bound), np.percentile(input_dom_im_backend, self.clip_upper_bound)
                else:
                    #Else, will use the positive voxels.
                    try:
                        lower_bound, upper_bound = np.percentile(input_dom_im_backend[input_dom_im_backend > 0], self.clip_lower_bound), np.percentile(input_dom_im_backend[input_dom_im_backend > 0], self.clip_upper_bound)
                    except:
                        lower_bound, upper_bound = 0, 0 #the case where there are no positive voxels, we just set the bounds to be 0,0.
                #Then we do as they do, we min-max on a global scale.
                input_dom_im_backend = np.clip(input_dom_im_backend, lower_bound, upper_bound)
                input_dom_im_backend = np.ceil((input_dom_im_backend - lower_bound) / (upper_bound - lower_bound + 1e-6) * 255).astype(np.uint8)
            # try:
            #     lower_bound, upper_bound = np.percentile(input_dom_im_backend[input_dom_im_backend > 0], self.clip_lower_bound), np.percentile(slice[slice > 0], self.clip_upper_bound) 
            # except:
            #     lower_bound, upper_bound = 0, 0
            
            for slice_idx in range(input_dom_im_backend.shape[ax]):
                if ax == 0:
                    slice = input_dom_im_backend[slice_idx, :, :]
                elif ax == 1:
                    slice = input_dom_im_backend[:, slice_idx, :]
                elif ax == 2:
                    slice = input_dom_im_backend[:, :, slice_idx]
                else:
                    raise Exception('Cannot have more than three spatial dimensions for indexing the slices, we only permit 3D volumes at most!')
                
                if not self.glob_norm_bool: #In this case we just try and do something very basic on a slice level..
                    #this is almost certainly not what one would consider "optimal" in the vast majority of cases.
                    if self.slice_norm_method == 'basic':
                        try:
                            lower_bound, upper_bound = np.percentile(slice[slice > 0], self.clip_lower_bound), np.percentile(slice[slice > 0], self.clip_upper_bound) 
                        except:
                            lower_bound, upper_bound = 0, 0
                            #In case that there was no foreground we just set the bounds to be 0,0.

                        #Clamping the voxel intensities.
                        slice = np.clip(slice, lower_bound, upper_bound) 
                        slice = np.round((slice - slice.min()) / (slice.max() - slice.min() + 1e-6) * 255).astype(
                            np.uint8
                        )  # Mapping to [0,255] rgb scale

                #We transpose the slice spatially, since RAS orientation does not align with the image array orientation of the demo, we assume self-consistency, this assumption may
                #not be valid but it is our only presumption given the provided information.
                slice = slice.T

                if self.sanity_check:
                    ax_slices_pre_resizing[slice_idx] = slice #We save this to help with our sanity checks.

                
                slice = np.repeat(slice[..., None], repeats=3, axis=-1) #RGB
                slice = (slice - self.pixel_mean) / self.pixel_std  # per-channel pixel normalisation according to the sam parameters.

                transforms = self.transforms(self.app_params['image_size'])
                augments = transforms(image=slice)
                slice = augments["image"].unsqueeze(0)  # Add batch dimension
                ax_slices_process[slice_idx] = slice.float() #Adding the slice array to the dictionary which will be fed forward.

            #Insertion into the dictionary of slices which contains sets of slices/affine planes orthogonal to the axis along which it was marching. 
            slices_processed[ax] = ax_slices_process 
            
            if self.sanity_check:
                self.im_slices_input_dom[ax] = ax_slices_pre_resizing 

            #saving of the original dimensions for the slice extracted, will be required for mapping back to the segmentation space.. #NOTE: Since we have transposed,
            #mapping the predictions back will first require transposing this shape, performing the inverse map, then transposition to return back to RAS space.
            orig_im_dims[ax] = np.array([input_dom_im_backend.shape[i] for i in set(list(range(input_dom_im_backend.ndim))) ^ set([ax])])
            assert orig_im_dims[ax].size == 2 and orig_im_dims[ax].ndim == 1
        return slices_processed, orig_im_dims
 
    @torch.no_grad()
    def binary_extract_im_embeddings(self, im_slices_model_dom:dict):
        #Function which is designed to extract the image embeddings of the set of input slices in the model domain, stores them for the entire duration of the iterative refinement
        #process. 
        
        for ax in self.app_params['image_axes']:
            axis_embeddings = dict() 
            for slice_idx, img_slice in im_slices_model_dom[ax].items():
            # with torch.no_grad():
                image_embedding = self.model.image_encoder(img_slice.to(self.infer_device))
                if slice_idx % self.cpu_gpu_burden_ratio:
                    axis_embeddings[slice_idx] = image_embedding.cpu().to(torch.float32)
                    #If not divisible by the burden ratio, we store on cpu. This will intrinsically favour the CPU so
                    #will require a hotfix later on to resolve this. 
                else:
                    axis_embeddings[slice_idx] = image_embedding.to(device=self.infer_device, dtype=torch.float32)
            self.image_embeddings_dict[ax] = axis_embeddings 

####################################################################################
#Functions for making the prediction given the input data which has been mapped to model domain.
    @torch.no_grad()
    def binary_predict(self, init_bool:bool, infer_slices: dict):
        #Function which takes the set of altered slices, the stored input information, and iterates through performing the predictions & updating the memory bank for
        #future iterations. It returns the dictionary of prob maps and dict of binary maps for each slice along each axial dimension as outputs.
        
        
        for ax in self.app_params['image_axes']:
            #We perform inference by axis (NOTE: for now since we have restricted this to being only a singular axis, any fusion across different axes is not implemented)
            if init_bool:
                self.internal_lowres_mask_storage[ax] = {slice_idx:None for slice_idx in range(len(self.image_embeddings_dict[ax]))}
                self.internal_discrete_output_mask_storage[ax] = {slice_idx:None for slice_idx in range(len(self.image_embeddings_dict[ax]))}
                self.internal_prob_output_mask_storage[ax] = {slice_idx:None for slice_idx in range(len(self.image_embeddings_dict[ax]))}
                #Initialising the set of discrete and probabilistic output masks which will be used for constructing the output volumes. We will use this to store the
                #results such that when editing iterations occur, it can be altered at the corresponding slice and then re-merged.

            #Now we actually perform inference!
            for slice_idx in infer_slices[ax]:

                slice_ps = self.model_prompts_storage_dict[ax][slice_idx]
                #Finding the available prompts, we can use the fact that a non available prompt is empty because we pre-filtered the background bboxes. 
                # Therefore, no ambiguity here. A bounding box that is available is indeed available (we already pre-deleted the background bboxes...)
                avail_ps = [k for k in self.permitted_prompts if slice_ps[k] != [] and slice_ps[f'{k}_labels'] != []]
                
                if avail_ps == []:
                    #No prompt info provided.. this is treated like autoseg.
                    if self.autoseg_infer:
                        #In the case where we configure it to actually attempt to perform autoseg inference, and then to use that for forward propagation.
                        logits_outputs, _, lowres_masks = self.binary_slice_predict(self.image_embeddings_dict[ax][slice_idx].to(device=self.infer_device), 
                                                                                    (None, None), (None, None), 
                                                                                    self.internal_lowres_mask_storage[ax][slice_idx], 
                                                                                    self.input_dom_shapes[ax][::-1].tolist(),
                                                                                    True)
                        #NOTE: Since we have transposed the slice extracted from the input image domain, mapping the predictions back to input domain will first 
                        # require transposing the shape of the slice, performing the inverse mapping wrt rescaling etc, transposition to back to input space, then
                        #combining all the slices together.

                        #Storing the lowres mask in memory. We follow the demo and convert this to a probabilistic map using sigmoid function.
                        self.internal_lowres_mask_storage[ax][slice_idx] = torch.sigmoid(lowres_masks).to(torch.float32)
                        #We keep these two separate by following the convention in the demo to use the lowres map for forward propagation.
                        prob_outputs = torch.sigmoid(logits_outputs).to(torch.float32)
                        discrete_outputs = (prob_outputs > self.mask_threshold_sigmoid).to(torch.uint8)
                    else:
                        #In the case where we do not actually perform autoseg inference, but just "skip over" and also pass a NoneType for future iterations such that
                        #the inference is not conditioned on a potentially sparse mask, but rather as though it is starting fresh.

                        #In this case, we return a tensor of -1000 for the output as this will eval to 0s at the floating point precision for the prob map under 
                        # sigmoid. Internally store a different mask variable, a Nonetype (i.e., it will treat that first interaction instance as an init..)
                        logits_outputs, lowres_masks = -1000 * torch.ones([1,1] + self.input_dom_shapes[ax][::-1].tolist()) , None
                        if not torch.all(torch.sigmoid(logits_outputs) == 0):
                            raise Exception('Error with the strategy for generating p = 0 maps.')
                        self.internal_lowres_mask_storage[ax][slice_idx] = lowres_masks  
                        #We keep these two separate by following the convention in the demo to use the lowres map for forward propagation.
                        prob_outputs = torch.sigmoid(logits_outputs).to(torch.float32) #.to(device=self.infer_device)
                        discrete_outputs = (prob_outputs > self.mask_threshold_sigmoid).to(torch.uint8)
                else:
                    #In this case we have prompts, we split our next operations between points & scribbles, and bboxes (as we treat scribbles as sets of points)
                    
                    if (slice_ps['points'] == [] or slice_ps['points_labels'] == []) and (slice_ps['scribbles'] == [] or slice_ps['scribbles_labels'] == []):
                        points_input = (None, None) 
                    else:
                        #At least one of the scribbles and points is valid
                        point_coors, point_lbs = [], []
                        for p in ['points', 'scribbles']:
                            if (slice_ps[p] != [] and slice_ps[f'{p}_labels'] != []):
                                point_coors.append(slice_ps[p])
                                point_lbs.append(slice_ps[f'{p}_labels'])
                        assert len(point_coors) != 0

                        #The points need to be in 1 x N_point x N_dim structure and lbs need to be in 1 x N_point structure.
                        points_input = (torch.cat(point_coors, dim=0)[None, :, :].to(device=self.infer_device), torch.cat(point_lbs, dim=0)[None, :].to(device=self.infer_device))

                    if slice_ps['bboxes'] == [] or slice_ps['bboxes_labels'] == []:
                        bboxes_input = (None, None)
                        multi_box_bool = None 
                    else:
                        assert slice_ps['bboxes'].shape[0] == slice_ps['bboxes_labels'].shape[0]
                        #In this case we have bbox prompts that are valid.
                        if slice_ps['bboxes_labels'].shape[0] == 1:
                            #In this case we only have one box in this slice.
                            multi_box_bool = False 
                        else:
                            #In this case we have more than one box in this slice. We will use this variable for unpacking the outputs appropriately.
                            multi_box_bool = True 
                            warnings.warn('The probabilistic map output for a multi-box process will be fused in a very naive manner, by taking the maximum of the probability maps voxelwise, note that the discrete prediction is however fused in the exact same manner as the demo (by discretising separately and then finding the union).')
                        bboxes_input = (slice_ps['bboxes'].to(device=self.infer_device), slice_ps['bboxes_labels'])
                        #Unlike the points, no modif needed for bbox, we already store these

                    #NOTE: Since we have transposed the slice extracted from the input image domain, mapping the predictions back to input domain will first 
                    # require reversing the ordered tuple denoting the shape of the slice, performing the inverse mapping wrt rescaling etc, transposition to back to input space, then
                    #combining all the slices together.
                    
                    if not self.split_forward_mask:
                        logits_outputs, _, lowres_masks = self.binary_slice_predict(self.image_embeddings_dict[ax][slice_idx].to(device=self.infer_device), 
                                                                                    points_input, bboxes_input, 
                                                                                    self.internal_lowres_mask_storage[ax][slice_idx], 
                                                                                    self.input_dom_shapes[ax][::-1].tolist(),
                                                                                    True)
                    else:
                        logits_outputs, _, lowres_masks = self.binary_slice_predict_modif(self.image_embeddings_dict[ax][slice_idx].to(device=self.infer_device), 
                                                                                    points_input, bboxes_input, 
                                                                                    self.internal_lowres_mask_storage[ax][slice_idx], 
                                                                                    self.input_dom_shapes[ax][::-1].tolist(),
                                                                                    True)
                    #NOTE: Very naive fusion strategy coming up for the probabilistic map, we take the max (because they're all supposed to be foreground of the same class). 
                    # For the discrete map we just take the union of the discrete maps generated for each bbox, following the convention of the demo.
                    
                    #We assume that the low res probabilistic map does not need any modification, and will just be forward propagated. This however will mean that 
                    #any deviation from the quantity of bbox at the "initialisation of the slice" will break as it requires an individualchannel 
                    # for each bbox. But this is consistent with the functionality provided.

                    self.internal_lowres_mask_storage[ax][slice_idx] = torch.sigmoid(lowres_masks).to(torch.float32) 

                    if multi_box_bool is not None and not multi_box_bool:
                        #Single box, pretty straight forward to evaluate this.
                        prob_outputs = torch.sigmoid(logits_outputs).to(torch.float32) #.to(device=self.infer_device)
                        discrete_outputs = (prob_outputs > self.mask_threshold_sigmoid).to(torch.uint8)
                    elif multi_box_bool is not None and multi_box_bool:
                        mask_dim = 0
                        if not logits_outputs.shape[mask_dim] > 1:
                            raise Exception(f'We implemented this wrong, the mask dimension from output indicates that 1 or fewer bboxes were used but we are in the handling for multiple bboxes')
                        #Multiple box, we take a naive approach for handling the probabilistic map output, we take max over all channels as it is a single foreground!
                        box_sep_prob_outputs = torch.sigmoid(logits_outputs).to(torch.float32)
                        discrete_outputs = (box_sep_prob_outputs > self.mask_threshold_sigmoid).to(torch.uint8)
                        #We reduce over the 0th dimension corresponding to the quantity of prompts which are treated as distinct object instances (i.e. for each bbox).
                        discrete_outputs = (discrete_outputs.sum(dim=mask_dim, keepdim=True) > 0).to(torch.uint8) #We sum over the mask dim, then binarise as we assume each instance is
                        #an instance of the given foreground class (and we are performing semantic segmentation)
                        
                        #Now we aggregate the probability map we want to output.
                        prob_outputs = box_sep_prob_outputs.max(dim=mask_dim, keepdim=True)[0] 
                        #.max returns a tuple of the tensor and the tensor of indices along that channel and where the max occurs (i.e. the argmax) 
                    elif multi_box_bool is None:
                        if slice_ps['bboxes'] != [] or slice_ps['bboxes_labels'] != []:
                            raise Exception('Should not have flagged box as being NoneType if there were boxes.')
                        #For non-box prompt types, we have little to worry about, it doesn't treat each prompt as a separate instance..
                        prob_outputs = torch.sigmoid(logits_outputs).to(torch.float32)
                        discrete_outputs = (prob_outputs > self.mask_threshold_sigmoid).to(torch.uint8)
            
                #Storing the output maps, first we check that the shapes are consistent with what is required, ESPECIALLY for the channel dimensions:
                #we reverse the list because the input dom shapes are extracted prior to the transposition required for mapping from RAS to y,x cv2 coordinates.
                if list(prob_outputs.shape) != [1, 1] + self.input_dom_shapes[ax][::-1].tolist() or list(discrete_outputs.shape) != [1, 1] + self.input_dom_shapes[ax][::-1].tolist():
                    raise Exception('The structure of the output discrete map, and output probability map (prior to undoing the transposition operation) should be identical to the input image spatial size.')
                self.internal_prob_output_mask_storage[ax][slice_idx] = prob_outputs[0,0,...]
                self.internal_discrete_output_mask_storage[ax][slice_idx] = discrete_outputs[0,0,...] 
                
                #Plotting the output as a sanity check.
                if slice_idx == self.sanity_slice_check:
                    sanity_check_output(prob_outputs[0,0,...], 'prob', self.orig_prompts_storage_dict[ax][slice_idx], False)
                    sanity_check_output(discrete_outputs[0,0,...], 'discrete', self.orig_prompts_storage_dict[ax][slice_idx], False)

            if any([val is None for val in self.internal_discrete_output_mask_storage[ax].values()]):
                raise Exception('We should not have a NoneType for any slice after performing inference with respect to the internal discrete mask storage')
            if any([val is None for val in self.internal_prob_output_mask_storage[ax].values()]):
                raise Exception('We should not have a NoneType for any slice after performing inference with respect to the internal probability mask storage')
            
    def binary_slice_predict(self, #This is essentially the original function provided in the demo code but slightly modified for flexibility with slight changes made to bbox structure.
                            im_embedding:torch.Tensor, 
                            points:tuple, 
                            bboxes:tuple, 
                            mask:torch.Tensor | None, 
                            original_size: tuple | list, #Original dimensions of the slice in the cv2 coordinates! (i.e. post-transposed from input dom slice.)
                            return_logits:bool = True,
                            ):
        #Function which takes the info for a single altered slice, and makes a prediction. Bboxes are either a tuple of Nones or an N_box x 4 tensor and N_box vector.
        #We only include the bbox label for the sake of checking that all bboxes are foreground, this should already have been handled.

        #Func is mostly taken from SAM-Med2D's predictor_sammed.py script.
        if points[0] is not None and bboxes[0] is not None:
            warnings.warn('This model was not trained on combinations of points and bounding boxes but was provided with them!')
        
        if points[0] is not None:
            points = points 
        else:
            points = None

        if mask is None:
            pass
        else:
            mask = mask.to(device=self.infer_device) 
            
        if bboxes[0] is not None and bboxes[0].shape[0] > 1:
            mask_list = [] #SAM Med2D treats each bbox as a separate entity entirely.
            # Embed prompts
            for i in range(bboxes[0].shape[0]):
                if bboxes[1][i] == 1: #SAM Med2D does not consider background bounding boxes as being meaningful. So only keep the foreground ones.
                    pre_boxes = bboxes[0][i:i+1,...]
                else:
                    raise Exception('Uncaught instance of background prompt.')  
                if mask is not None:
                    if mask.shape[0] != 1:
                        raise Exception('Cannot have more than one mask in the forward propagated mask, if a mask is being forward propagated.')   
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=pre_boxes,
                    masks=mask,
                )

                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=im_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=self.multimask_output_always,
                )

                if self.multimask_output_always:
                    max_values, max_indexs = torch.max(iou_predictions, dim=1) #This is taking the maximum across the mask channel, not the batch! So the multi-box is not being affected.
                    max_values = max_values.unsqueeze(1)
                    iou_predictions = max_values
                    low_res_masks = low_res_masks[:, max_indexs]

                # Upscale the masks to the original image resolution
                masks = self.postprocess_masks(low_res_masks, self.model.image_encoder.img_size, original_size)
        
                mask_list.append(masks)

            masks = torch.cat(mask_list, dim=0)
            
            del mask_list 

        else: #In the case where we either don't have a bbox, or we only have one the post processing is the same, so we group them together.
            # Embed prompts
            if bboxes[0] is not None and bboxes[0].shape[0] == 1:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=bboxes[0],
                    masks=mask,
                )
            else:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=mask,
                )

            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=im_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output_always,
            )

            if self.multimask_output_always: #This extracts the mask according to the highest predicted iou
                max_values, max_indexs = torch.max(iou_predictions, dim=1)
                max_values = max_values.unsqueeze(1)
                iou_predictions = max_values
                low_res_masks = low_res_masks[:, max_indexs]

            # Upscale the masks to the original image resolution
            masks = self.postprocess_masks(low_res_masks, self.model.image_encoder.img_size, original_size)

        if not return_logits:
            sigmoid_output = torch.sigmoid(masks)
            masks = (sigmoid_output > self.mask_threshold_sigmoid).float()

        if bboxes != (None, None):
            assert masks.shape[0] == bboxes[0].shape[0] 
            assert low_res_masks.shape[0] == 1 
        else:
            assert masks.shape[0] == 1 
            assert low_res_masks.shape[0] == 1
        
        #HACK: to help prevent segfaulting of the GPU VRAM.
        mask = mask.to(device='cpu')
        low_res_masks = low_res_masks.to(device='cpu')
        masks = masks.to(device='cpu')
        sparse_embeddings = sparse_embeddings.to(device='cpu')
        dense_embeddings = dense_embeddings.to(device='cpu')
        gc.collect()
        torch.cuda.empty_cache() 

        return masks, iou_predictions, low_res_masks
    

    def binary_slice_predict_modif(self, 
                        im_embedding:torch.Tensor, 
                        points:tuple, 
                        bboxes:tuple, 
                        mask:torch.Tensor | None, 
                        original_size: tuple | list, #Original dimensions of the slice in the cv2 coordinates! (i.e. post-transposed from input dom slice.)
                        return_logits:bool = True,
                        ):
        #Function which takes the info for a single altered slice, and makes a prediction. Bboxes are either a tuple of Nones or an N_box x 4 tensor and N_box vector.
        #We only include the bbox label for the sake of checking that all bboxes are foreground, this should already have been handled.

        #This is a modification made to the mostly original version of this function, which is not adequately flexible for instances where we are iterating over
        # a segmentation with n_box > 1. The original function will not output the corresponding lowres_masks for each box (corresponding to what they do for the
        # logits/predictions) and cannot handle cases where the number of prior masks > 1 also (which can occur if #bbox > 1 in a slice), instead it can only use a single
        # prior mask which may not correspond to the given bbox used (as it only outputs the last lowres mask from the list of bboxes for storage).

        if points[0] is not None and bboxes[0] is not None:
            warnings.warn('This model was not trained on combinations of points and bounding boxes but was provided with them!')
        
        if points[0] is not None:
            points = points 
        else:
            points = None

        if mask is None:
            pass
        else:
            mask = mask.to(device=self.infer_device) 

        if bboxes[0] is not None and bboxes[0].shape[0] > 1:
            mask_list = [] #SAM Med2D treats each bbox as a separate entity entirely.
            #We add a way of tracking each of the lowres masks for forward propagation:
            lowres_masks_list = [] 

            # Embed prompts
            for i in range(bboxes[0].shape[0]):
                if bboxes[1][i] == 1: #SAM Med2D does not consider background bounding boxes as being meaningful. So only keep the foreground ones.
                    pre_boxes = bboxes[0][i:i+1,...]
                else:
                    raise Exception('Uncaught instance of background prompt.')     
                
                if mask is None:
                    #If nonetype, then proceed as normal! We have no prior prediction for this slice.
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=points,
                        boxes=pre_boxes,
                        masks=mask
                    ) 
                elif isinstance(mask, torch.Tensor):
                    if mask.shape[0] == 1: 
                        #In this case, the prediction could have been initialised by another prompt, like points or scribbles, prior to the introduction of a bbox.
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=points,
                            boxes=pre_boxes,
                            masks=mask.to(device=self.infer_device),
                        )
                    elif mask.shape[0] == bboxes[0].shape[0]:
                        #In the case where we have multiple masks and so need to select the appropriate one.
                        assert mask.shape[1] == 1 #Assert that the channel dim is 1.
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=points,
                            boxes=pre_boxes,
                            masks=mask[i,...].unsqueeze(dim=0),
                        )
                    elif mask.shape[0] != bboxes[0].shape[0]:
                        #This scenario shouldn't happen/isn't supported. There is no way to map the set of N masks to M bboxes in any meaningful way.
                        raise Exception('Should not be a mismatch between the quantity of masks and the quantity of bboxes which can only occur due to a change in number of bboxes, unless it previously had 1 (due to non-bbox or autoseg initialisation)')


                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=im_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=self.multimask_output_always,
                )

                if self.multimask_output_always:
                    max_values, max_indexs = torch.max(iou_predictions, dim=1) #This is taking the maximum across the mask channel, not the batch! So the multi-box is not being affected.
                    max_values = max_values.unsqueeze(1)
                    iou_predictions = max_values
                    low_res_masks = low_res_masks[:, max_indexs]

                # Upscale the masks to the original image resolution
                processed_masks = self.postprocess_masks(low_res_masks, self.model.image_encoder.img_size, original_size)
        
                mask_list.append(processed_masks)
                lowres_masks_list.append(low_res_masks) 

            masks = torch.cat(mask_list, dim=0)
            low_res_masks = torch.cat(lowres_masks_list, dim=0) #We add this for forward propagation of the entire set of the low res masks.
            assert masks.shape[0] == low_res_masks.shape[0] == bboxes[0].shape[0]
            del mask_list 
            del lowres_masks_list

        else:
            #In any case where bboxes are not provided or we only have one bbox, we only have a singular set of output masks, can take the standard approach.
            # Embed prompts

            if bboxes[0] is not None and bboxes[0].shape[0] == 1:
                if not bboxes[1][0] == 1: #SAM Med2D does not consider background bounding boxes as being meaningful. So should only keep the foreground ones.
                    raise Exception('Uncaught instance of background prompt.')     
                
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=bboxes[0],
                    masks=mask,
                )
            else:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=mask,
                )

            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=im_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output_always,
            )

            if self.multimask_output_always: #This extracts the mask according to the highest predicted iou
                max_values, max_indexs = torch.max(iou_predictions, dim=1)
                max_values = max_values.unsqueeze(1)
                iou_predictions = max_values
                low_res_masks = low_res_masks[:, max_indexs]

            # Upscale the masks to the original image resolution
            masks = self.postprocess_masks(low_res_masks, self.model.image_encoder.img_size, original_size)
        
        if not return_logits:
            sigmoid_output = torch.sigmoid(masks)
            masks = (sigmoid_output > self.mask_threshold_sigmoid).float()


        if bboxes != (None, None):
            assert masks.shape[0] == bboxes[0].shape[0] 
            assert low_res_masks.shape[0] == bboxes[0].shape[0] 
        else:
            assert masks.shape[0] == 1 
            assert low_res_masks.shape[0] == 1

        #HACK: to help prevent segfaulting of the GPU VRAM.
        if mask is None:
            pass 
        else:
            mask = mask.to(device='cpu')
        low_res_masks = low_res_masks.to(device='cpu')
        masks = masks.to(device='cpu')
        sparse_embeddings = sparse_embeddings.to(device='cpu')
        dense_embeddings = dense_embeddings.to(device='cpu')
        gc.collect()
        torch.cuda.empty_cache() 

        return masks, iou_predictions, low_res_masks
    



############################################################

    def binary_merge_slices(self):
        """
        Slice merging steps:

            - Combine inferred slices into one volume while also undoing the transposition according to the axes (swapping the channels) such that the output 
            matches the RAS structure of the input image.
            
                This is done due to the fact that the input image in RAS would not be aligned with the coordinates used by cv2.
        """
        #We assert that the quantity of axes once again must be 1, since there is not an obvious way to merge slices otherwise with a concatenation.

        if not len(self.app_params['image_axes']) == 1:
            raise Exception('Cannot merge together slices in multiple axes with a simple concatenation')
        

        for ax in self.app_params['image_axes']:
            merged_discrete = torch.cat([(mask.T).unsqueeze(dim=ax) for mask in self.internal_discrete_output_mask_storage[ax].values()], dim=ax)
            merged_prob = torch.cat([(mask.T).unsqueeze(dim=ax) for mask in self.internal_prob_output_mask_storage[ax].values()], dim=ax)
            
            assert self.orig_im_shape == tuple(merged_discrete.shape)
            assert self.orig_im_shape == tuple(merged_prob.shape)

        #Now we channel-split the probability map by class.
        prob_map_list = []
        for label in self.configs_labels_dict.keys():
            if label.title() == 'Background':
                prob_map_list.append(1 - merged_prob)
            else:
                prob_map_list.append(merged_prob)
        merged_prob = torch.stack(prob_map_list, dim=0)
        merged_discrete = merged_discrete.unsqueeze(dim=0)
        
        assert merged_discrete.ndim == 4
        assert merged_prob.ndim == 4
        assert merged_prob.shape[0] == len(self.configs_labels_dict)

        return merged_discrete, merged_prob #merged_prob.unsqueeze(dim=0)

########################################################
    def transforms(self, new_size):  
        #Copied from SAM-Med2D predictor_sammed.py
        Transforms = []
        new_h, new_w = new_size
        Transforms.append(
            A.Resize(int(new_h), int(new_w), interpolation=cv2.INTER_NEAREST)
        )  # note nearest neighbour interpolation.
        Transforms.append(ToTensorV2(p=1.0))
        return A.Compose(Transforms, p=1.0)

    def apply_coords(self, coords, original_size, new_size):
        #copied from SAM-Med2D predictor_sammed.py
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = copy.deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        #Copied from SAM-Med2D predictor_sammed.py
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)
    
    def postprocess_masks(self, low_res_masks, image_size, original_size):
        ori_h, ori_w = original_size
        masks = F.interpolate(low_res_masks,(image_size, image_size), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


    def __call__(self, request:dict):

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError 
        else:
            raise Exception('Should not have received less than two class labels at minimum')
        
        #We create a duplicate so we can transform the data from metatensor format to the torch tensor format compatible with the inference script.
        modif_request = copy.deepcopy(request) 

        app = self.infer_apps[modif_request['model']][f'{class_type}_predict']

        #Setting the configs label dictionary for this inference request.
        self.configs_labels_dict = modif_request['config_labels_dict']

        pred, probs_tensor, affine = app(request=modif_request)

        pred = pred.to(device='cpu')
        probs_tensor = probs_tensor.to(device='cpu')
        affine = affine.to(device='cpu')
        torch.cuda.empty_cache()


        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['metatensor'].meta['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor,
                'meta_dict':{'affine': affine}
            },
            'pred':{
                'metatensor':pred,
                'meta_dict':{'affine': affine}
            },
        }
        #Functionally probably wont do anything but putting it here as a placebo. Won't make a diff because there are references
        #to these variables throughout.
        del pred 
        del probs_tensor
        del affine
        del modif_request
        gc.collect() 
        return output 
    
if __name__ == '__main__':
   
    infer_app = InferApp(
        {'dataset_name':'BraTS2021',
        'dataset_modality':'MRI'}, torch.device('cuda', index=0))

    infer_app.app_configs()

    from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose 
    import nibabel as nib 

    input_dict = {'image':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTs/BraTS2021_00266.nii.gz'}
    load_and_transf = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])

    final_loaded_im = load_and_transf(input_dict)
    meta = {'original_affine': torch.from_numpy(final_loaded_im['image_meta_dict']['original_affine']).to(dtype=torch.float64), 'affine': torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64)}
    input_metatensor = MetaTensor(x=torch.from_numpy(final_loaded_im['image']).to(dtype=torch.float64), meta=meta) #affine=torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64))
    # MetaTensor(x=torch.from_numpy(final_loaded_im['image']).to(dtype=torch.float64), meta=final_loaded_im['image_meta_dict'], affine=torch.from_numpy(final_loaded_im['image_meta_dict']['affine']).to(dtype=torch.float64))
    request = {
        'image':{
            'metatensor': input_metatensor,
            'meta_dict':{'affine':input_metatensor.affine}
        },
        'model':'IS_interactive_init',
        # 'model': 'IS_autoseg',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        # {'Automatic Init': None,
        {'Interactive Init':{
            'interaction_torch_format': {
                'interactions': {
                    'points': [torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]]), torch.tensor([[61, 62, 39]])], #None
                    'scribbles': None,#[torch.tensor([[63,62,39], [64,62,39],[65,62,39]]), torch.tensor([[73,62,39], [74,62,39],[75,62,39]])], 
                    #This second scribble is a fugazi but intended just for sanity checking the mapping.
                    'bboxes': None#[torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64), torch.Tensor([[93,80,30, 105, 100, 51]]).to(dtype=torch.int64)]  #None 
                    },#This second box is a fugazi but intended just for sanity checking that the multi-box method works. 17-51 should be the real bounds.
                'interactions_labels': {
                    'points_labels': [torch.tensor([0]), torch.tensor([1]), torch.tensor([1])], #None,#[torch.tensor([0]), torch.tensor([1])], 
                    'scribbles_labels':None,#[torch.tensor([1]), torch.tensor([0])],  
                    'bboxes_labels': None#[torch.Tensor([1]).to(dtype=torch.int64), torch.Tensor([1]).to(dtype=torch.int64)] #None
                    },
                }
            # 'interaction_dict_format': {
            # 'points': {'background': [[40, 103, 43]],
            # 'tumor': [[61,62,39],[62, 62, 39]]
            # },
            # 'points': None,
            # 'scribbles': None, #{'background':[[73,62,39], [74,62,39],[75,62,39]]], 'tumor': [[[63,62,39], [64,62,39], [65,62,39]]] }
            # 'bboxes': None, #{'background': [], 'tumor': [[56,30,17, 92, 76, 51]]} #None
            },
            # 'prev_probs': {'metatensor': None, 'meta_dict': None}, 
            # 'prev_pred': {'metatensor': None, 'meta_dict': None}}
        },
    }
    output = infer_app(request)
    # print('halt')



    request2 = {
        'image':{
            'metatensor': input_metatensor,
            'meta_dict':{'affine':input_metatensor.affine}
        },
        'model':'IS_interactive_edit',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        {'Interactive Edit Iter 1':{
            'interaction_torch_format': {
                'interactions': {
                    'points': [torch.tensor([[62,62,39]])],#[torch.tensor([[41, 103, 69]]), torch.tensor([[62, 62, 57]]), torch.tensor([[61, 62, 57]])], #None
                    'scribbles': None,#[torch.tensor([[63,62,29], [64,62,29],[65,62,29]]), torch.tensor([[73,62,57], [74,62,57],[75,62,57]])], 
                    #This second scribble is a fugazi but intended just for sanity checking the mapping.
                    'bboxes': [torch.Tensor([[56,30,17, 92, 76, 39]]).to(dtype=torch.int64), torch.Tensor([[93,80,30, 105, 100, 51]]).to(dtype=torch.int64)]  #None 
                    },#This second box is a fugazi but intended just for sanity checking that the multi-box method works. 17-51 should be the real bounds.
                'interactions_labels': {
                    'points_labels': [torch.tensor([1])],#[torch.tensor([0]), torch.tensor([1]), torch.tensor([1])], #None,#[torch.tensor([0]), torch.tensor([1])], 
                    'scribbles_labels':None,#[torch.tensor([1]), torch.tensor([0])],  
                    'bboxes_labels': [torch.Tensor([1]).to(dtype=torch.int64), torch.Tensor([1]).to(dtype=torch.int64)] #None
                    }
                    },
          
            # 'interaction_dict_format': {
            # 'points': {'background': [[40, 103, 43]],
            # 'tumor': [[61,62,39],[62, 62, 39]]
            # },
            # # 'points': None,
            # 'scribbles': None, #{'background':[[73,62,39], [74,62,39],[75,62,39]]], 'tumor': [[[63,62,39], [64,62,39], [65,62,39]]] }
            # 'bboxes': None, #{'background': [], 'tumor': [[56,30,17, 92, 76, 51]]} #None
            # },
            'prev_probs': {'metatensor': None, 'meta_dict': None}, 
            'prev_pred': {'metatensor': None, 'meta_dict': None}}
        },
    }

    output2 = infer_app(request2)
    print('halt')