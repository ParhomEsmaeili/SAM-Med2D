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

import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from segment_anything import sam_model_registry as registry_sammed2d
# from radioa.prompts.prompt import PromptStep
# from radioa.model.inferer import Inferer
# from radioa.utils.transforms import orig_to_SAR_dense, orig_to_canonical_sparse_coords
# from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib

########################################
from monai.data import MetaTensor 
import re

def load_sammed2d(checkpoint_path, encoder_model_type, image_size, device="cuda"):
    args = Namespace()
    args.image_size = image_size
    args.encoder_adapter = True
    args.sam_checkpoint = checkpoint_path
    model = registry_sammed2d[encoder_model_type](args).to(device)
    model.eval()

    return model


class InferApp:
    # pass_prev_prompts = True  # Flag to track whether in interactive steps previous prompts should be passed, or only the mask and the new prompt
    # dim = 2
    # supported_prompts: Sequence[str] = ("box", "point", "mask")
    # transform_reverses_order = True
    def __init__(self, dataset_info, infer_device):
        
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

        #Initialising any remaining variables required for performing inference.

        self.logit_threshold = 0  # Hardcoded

        self.image_embeddings_dict = {}
        self.multimask_output = True  # Hardcoded to match defaults from original

        self.pixel_mean, self.pixel_std = (
            self.model.pixel_mean.squeeze().cpu().numpy(),
            self.model.pixel_std.squeeze().cpu().numpy(),
        )

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
            'IS_autoseg':{'binary_predict':self.binary_predict},
            'IS_interactive_init': {'binary_predict':self.binary_predict},
            'IS_interactive_edit': {'binary_predict':self.binary_predict}
            }
    





    @torch.no_grad()
    def binary_predict(self, request):
        #Mapping the input request to the model domain:
        mapped_inputs = self.binary_subject_prep(request=request)

        #Performing inference on the mapped inputs, and using any prior retained knowledge.


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
            key = 'Interactive Init' 
            is_state = request['im'][key]
            init_im_embeddings = False 
            
            assert self.image_embeddings
            assert self.model_dom_masks

        elif request['model'] == 'IS_interactive_init':
            key = 'Interactive Init' 
            is_state = request['im'][key]
            init_im_embeddings = True 

            self.image_embeddings = {}
            self.model_dom_masks = {} 

        elif request['model'] == 'IS_autoseg':
            key = 'Automatic Init'
            is_state = request['im'][key]
            if is_state is not None:
                raise Exception('Autoseg should not have any interaction info.')
            init_im_embeddings = True 

            self.image_embeddings = {} 
            self.model_dom_masks = {} 

        #Extracting the image in the model's coordinate space. NOTE: In order to disentangle the validation framework from inference apps 
        # this is always assumed to be handled within the inference app.

        mapped_input = self.binary_prop_to_model(request['image'], is_state, init_im_embeddings)        

        return mapped_input 

    
    def binary_prop_to_model(self, im_dict: dict, is_state: dict | None, init_im_embed: bool):
        
        #We first propagate the image into the model domain, and store the image embeddings.

        if init_im_embed:
            input_dom_img = im_dict['metatensor']
            input_dom_affine = im_dict['meta_dict']['affine']
            # input_dom_shape = input_dom_img.shape[1:] #Assuming a channel-first image is being provided.
            im_slices_model_dom, input_dom_shapes = self.im_to_model_dom(input_dom_img) 

            #Now we will extracted the image embeddings correspondingly, and store them in memory.



        #Now we propagate the prompt information into the model domain. 

        if bool(is_state):
            #Placing the prompts into a tensor, we will be doing this in the same capacity as the demo implementation of SegVol which will inevitably lead to 
            # information loss.
            p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
            
            coords = labels = input_p_mask = None
            
            #Determine the prompt type from the input prompt dictionaries: Not sure if intersection is optimal for catching exceptions here.
            provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
            if not len(provided_ptypes) == 1:
                raise Exception(f'Only one prompt is permitted for SegVol when using zoom-in activated, we received {len(provided_ptypes)}')
            
            if provided_ptypes[0] == "points":
                
                #NOTE: The strategy employed by SegVol when working with image representations of prompt inputs will inevitably lead to the deletion of background prompts 
                # as they only retain the 1s. (Whatever that is depends on the definition here, but typically it will be some arbitary foreground.)
                coords = torch.cat(p_dict[0]['points'], dim=0)
                labels = torch.cat(p_dict[1]['points_labels'], dim=0)
                # points_input = (coords.unsqueeze(0).to(device=self.infer_device), labels.unsqueeze(0).to(device=self.infer_device))
                input_p_mask = build_binary_points(coords, labels, input_dom_shape).unsqueeze(0)
                input_p_mask = input_p_mask

            elif provided_ptypes[0] == "bboxes":
                #NOTE: The strategy employed by SegVol when working with image representations of prompt inputs will inevitably lead to the deletion of background prompts 
                # as they only retain the 1s. (Whatever that is depends on the definition here, but typically it will be some arbitary foreground.)
                #NOTE: We can typically assume that the background probably won't have a bbox because that doesn't really have an inherent meaning.... 

                coords = torch.cat(p_dict[0]['bboxes'], dim=0)
                labels = torch.stack(p_dict[1]['bboxes_labels'])

                #Extracting the set of coordinate info by picking only the foreground bbox as segvol does.
                idxs = torch.argwhere(labels == 1)[:,0].tolist()
                input_bbox = coords[idxs, :]
                if input_bbox.shape[0] > 1:
                    raise Exception('Cannot handle more than one foreground bounding box at a given time.')
                elif input_bbox.shape[0] == 0:
                    warnings.warn('There was no foreground bounding box provided for this given class (class=foreground if binary segmentation task.)')
                    input_p_mask = torch.zeros_like(input_dom_img)
                else:
                    #Creating the image array representation if we have one bbox!
                    input_p_mask = build_binary_cube(input_bbox, input_dom_shape).unsqueeze(0)
        
            else:
                raise Exception('No other prompting types are supported in SegVol.')

            if input_p_mask is None:
                raise Exception('BUG: Prompt mask was not generated despite the fact that there was a valid input prompt, even if it was empty due to handling of binary classes..') 

        else:
            #Handling empty prompt dict and/or Autosegmentation.
            input_p_mask = torch.zeros_like(input_dom_img)
            provided_ptypes = [None]
                
        (img_fg_dom, img_zoomout_dom), (prompt_fg_dom, prompt_zoomout_dom), (start_coord, end_coord) = self.input_forward_map(
            input_dom_img, input_p_mask
        )    
        return {
            'im_slices_model_dom': im_slices_model_dom,
            'prompt_model_dom': prompt_fg_dom, 
            'input_dom_affine': input_dom_affine,
            'input_dom_shapes': input_dom_shapes, #This is the dictionary, for each axis which we extract slices for, that contains the corresponding size of the image slices.
        }
    


    def transform_to_model_coords_dense(self, nifti: str | Path | nib.Nifti1Image, is_seg: bool) -> np.ndarray:
        # Model space is always throughplane first (commonly the z-axis)
        data, inv_trans = orig_to_SAR_dense(nifti)

        return data, inv_trans

    def transform_to_model_coords_sparse(self, coords: np.ndarray) -> np.ndarray:
        return orig_to_canonical_sparse_coords(coords, self.orig_affine, self.orig_shape)

    def segment(self, points, box, mask, image_embedding):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )

        if self.multimask_output:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            low_res_masks = low_res_masks[:, max_indexs]

        return low_res_masks


    def transforms(self, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        Transforms = []
        new_h, new_w = new_size
        Transforms.append(
            A.Resize(int(new_h), int(new_w), interpolation=cv2.INTER_NEAREST)
        )  # note nearest neighbour interpolation.
        Transforms.append(ToTensorV2(p=1.0))
        return A.Compose(Transforms, p=1.0)

    def apply_coords(self, coords, original_size, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 2] = coords[..., 2] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def apply_boxes(self, boxes, original_size, new_size):  # Copied over from SAM-Med2D predictor_sammed.py
        boxes = self.apply_coords(boxes.reshape(2, 3), original_size, new_size)
        boxes = boxes[:, 1:]  # Remove z coord
        return boxes.reshape(-1, 4)

    def im_to_model_dom(self, input_dom_im): #Mostly borrowed from RadioActive SAM-Med2D inferer.
        #Assuming that input image is in RAS convention, the axes denote the subset from (0,1,2) which denotes the axes along which slices will be taken, i.e., 
        #if (2), then the values being extracted are from the first 2 according to the third index (i.e. in the R-L/A-I plane, aka axial slices)
        
        #First removing the channel dimension and converting to a numpy array.
        input_dom_im_backend = copy.deepcopy(input_dom_im).data.numpy()[0,:]

        slices_processed = {}
        orig_im_dims = {}

        if len(self.app_params['image_axes']) > 1:
            raise Exception('Implementation currently is not capable of simultaneous handling of > 1 planar segmentations.')
        for ax in self.app_params['image_axes']:
            
            ax_slices_process = {} 
            #This normalisation logic is borrowed from RadioActive, as SAM-Med2D does not provide their own preprocessing scripts aside from what is assumed for 
            # SAM (we assume this is done externally). The logic is fairly standard, and not specialised for specific datasets. This will also mean that the fg voxels
            #will probably be normalised in a manner where mean != 0, since the normalisation parameters are computed on the foregrounds only!
            for slice_idx in range(input_dom_im_backend.shape[ax]):
                if ax == 0:
                    slice = input_dom_im_backend[slice_idx, :, :]
                elif ax == 1:
                    slice = input_dom_im_backend[:, slice_idx, :]
                elif ax == 2:
                    slice = input_dom_im_backend[:, :, slice_idx]
                else:
                    raise Exception('Cannot have more than three spatial dimensions for indexing the slices, we only permit 3D volumes at most!')
                try:
                    lower_bound, upper_bound = np.percentile(slice[slice > 0], 0.5), np.percentile(slice[slice > 0], 99.5) 
                except:
                    lower_bound, upper_bound = 0, 0
                    
                slice = np.clip(slice, lower_bound, upper_bound)
                slice = np.round((slice - slice.min()) / (slice.max() - slice.min() + 1e-6) * 255).astype(
                    np.uint8
                )  # Get slice into [0,255] rgb scale
                slice = np.repeat(slice[..., None], repeats=3, axis=-1)  # Add channel dimension to make it RGB-like
                slice = (slice - self.pixel_mean) / self.pixel_std  # normalise

                transforms = self.transforms(self.app_params['image_size'])
                augments = transforms(image=slice)
                slice = augments["image"][None, :, :, :]  # Add batch dimension

                ax_slices_process[slice_idx] = slice.float()
            #Insertion into the dictionary of slices for that given plane, marched along that axis.
            slices_processed[ax] = ax_slices_process 
            #saving of the original dimensions for the planar extracted/slice extracted.
            orig_im_dims[ax] = np.array([input_dom_im_backend.shape[i] for i in set(list(range(input_dom_im_backend.ndim))) ^ set([ax])])
        return slices_processed, orig_im_dims


    def preprocess_prompt(self, prompt, promptstep_in_model_coord_system=False):
        """
        Preprocessing steps:
            - Modify in line with the volume cropping
            - Modify in line with the interpolation
            - Collect into a dictionary of slice:slice prompt
        """

        preprocessed_prompts_dict = {
            slice_idx: {"point": None, "box": None} for slice_idx in prompt.get_slices_to_infer()
        }

        if prompt.has_points:
            coords, labs = prompt.coords, prompt.labels
            coords, labs = np.array(coords).astype(float), np.array(labs).astype(int)

            # coords = coords[:,[2,1,0]] # Change from ZYX to XYZ
            coords_resized = self.apply_coords(coords, (self.H, self.W), self.new_size)

            # Convert to torch tensor
            coords_resized = torch.as_tensor(coords_resized, dtype=torch.float)
            labs = torch.as_tensor(labs, dtype=int)

            # Collate
            for slice_idx in prompt.get_slices_to_infer():
                slice_coords_mask = coords_resized[:, 0] == slice_idx
                slice_coords, slice_labs = (  # Subset to slice
                    coords_resized[slice_coords_mask],
                    labs[slice_coords_mask],
                )
                slice_coords = slice_coords[:, [2, 1]]  # leave out z and reorder
                slice_coords, slice_labs = slice_coords.unsqueeze(0).to(self.device), slice_labs.unsqueeze(0).to(
                    self.device
                )  # add batch dimension, move to device.
                preprocessed_prompts_dict[slice_idx]["point"] = (slice_coords, slice_labs)

        if prompt.has_boxes:
            for slice_index, box in prompt.boxes.items():
                box = np.array(
                    [slice_index, box[1], box[0], slice_index, box[3], box[2]]
                )  # Transform reverses coordinates, so desired points must be given as zyx
                box = self.apply_boxes(box, (self.H, self.W), self.new_size)
                box = np.array([box[0, 1], box[0, 0], box[0, 3], box[0, 2]])[None]  # Desperate fix attempt
                box = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box = box[None, :]
                preprocessed_prompts_dict[slice_index]["box"] = box.to(self.device)

        return preprocessed_prompts_dict

    def postprocess_slices(self, slice_mask_dict, return_logits):
        """
        Postprocessing steps:
            - Combine inferred slices into one volume, interpolating back to the original volume size
            - Turn logits into binary mask
            - Invert crop/pad to get back to original image dimensions
        """
        # Combine segmented slices into a volume with 0s for non-segmented slices

        dtype = np.float32 if return_logits else np.uint8
        segmentation = np.zeros((self.D, self.H, self.W), dtype)

        for z, low_res_mask in slice_mask_dict.items():
            mask = F.interpolate(low_res_mask, self.new_size, mode="bilinear", align_corners=False)
            mask = F.interpolate(
                mask, self.original_size, mode="bilinear", align_corners=False
            )  # upscale in two steps to match original code

            mask = torch.sigmoid(mask)
            if not return_logits:
                mask = (mask > 0.5).to(torch.uint8)

            segmentation[z, :, :] = mask.cpu().numpy()

        return segmentation

    @torch.no_grad()
    def predict(
        self, prompt, return_logits=False, prev_seg=None, promptstep_in_model_coord_system=False
    ) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
        if not (isinstance(prompt, PromptStep)):
            raise TypeError(f"Prompts must be supplied as an instance of the Prompt class.")
        if prompt.has_boxes and prompt.has_points:
            logger.warning("Both point and box prompts have been supplied; the model has not been trained on this.")

        if self.loaded_image is None:
            raise RuntimeError("Need to set an image to predict on!")

        prompt = deepcopy(prompt)
        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)
        slices_to_infer = prompt.get_slices_to_infer()

        self.D, self.H, self.W = self.img.shape
        self.original_size = (self.H, self.W)

        mask_dict = prompt.masks if prompt.masks is not None else {}
        preprocessed_prompt_dict = self.preprocess_prompt(prompt)
        slices_to_process = [
            slice_idx for slice_idx in slices_to_infer if slice_idx not in self.image_embeddings_dict.keys()
        ]

        slices_processed = self.preprocess_img(self.img, slices_to_process)

        self.slice_lowres_outputs = {}
        for slice_idx in slices_to_infer:
            # Get image embedding (either create it, or read it if stored and desired)
            if slice_idx in self.image_embeddings_dict.keys():
                image_embedding = self.image_embeddings_dict[slice_idx].to(self.device)
            else:
                slice = slices_processed[slice_idx]
                with torch.no_grad():
                    image_embedding = self.model.image_encoder(slice.to(self.device))
                self.image_embeddings_dict[slice_idx] = image_embedding.cpu()

            # Get prompts
            slice_points, slice_box = (
                preprocessed_prompt_dict[slice_idx]["point"],
                preprocessed_prompt_dict[slice_idx]["box"],
            )
            slice_mask = (
                torch.from_numpy(mask_dict[slice_idx]).to(self.device).unsqueeze(0).unsqueeze(0)
                if slice_idx in mask_dict.keys()
                else None
            )

            # Infer
            slice_raw_outputs = self.segment(
                points=slice_points, box=slice_box, mask=slice_mask, image_embedding=image_embedding
            )
            self.slice_lowres_outputs[slice_idx] = slice_raw_outputs

        low_res_logits = {k: torch.sigmoid(v).squeeze().cpu().numpy() for k, v in self.slice_lowres_outputs.items()}

        segmentation = self.postprocess_slices(self.slice_lowres_outputs, return_logits)

        # Fill in missing slices using a previous segmentation if desired
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        # Reorient to original orientation and return with metadata
        # Turn into Nifti object in original space
        # Turn into Nifti object in original space
        segmentation_model_arr = segmentation
        segmentation_orig_nib = self.inv_trans_dense(segmentation)

        return segmentation_orig_nib, low_res_logits, segmentation_model_arr


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


        probs_tensor, pred, affine = app(request=modif_request)




        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['metatensor'].meta['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
            'pred':{
                'metatensor':pred.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
        }
        return output 
    
if __name__ == '__main__':
   
    infer_app = InferApp(
        {'dataset_name':'BraTS2021',
        'dataset_modality':'MRI'}, torch.device('cuda'))

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
        # 'model':'IS_interactive_edit',
        'model': 'IS_interactive_init',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        
        # {'Automatic Init': None}
        {'Interactive Init':{
            'interaction_torch_format': {
                'interactions': {
                    'points': [torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]])], #None
                    'scribbles': None, 
                    'bboxes': None, #[torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64)] #None 
                    },
                'interactions_labels': {
                    'points_labels': [torch.tensor([0]), torch.tensor([1])], #None,#[torch.tensor([0]), torch.tensor([1])], 
                    'scribbles_labels': None, 
                    'bboxes_labels': None #[torch.Tensor([1]).to(dtype=torch.int64)] #None
                    }
                    },
          
            'interaction_dict_format': {
            'points': {'background': [[40, 103, 43]],
            'tumor': [[62, 62, 39]]
            },
            # 'points': None,
            'scribbles': None,
            'bboxes': None, #{'background': [], 'tumor': [[56,30,17, 92, 76, 51]]} #None
            },
            'prev_probs': {'metatensor': None, 'meta_dict': None}, 
            'prev_pred': {'metatensor': None, 'meta_dict': None}}
        },
    }
    output = infer_app(request)
    print('halt')