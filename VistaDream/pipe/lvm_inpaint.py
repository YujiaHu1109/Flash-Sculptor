'''
render using frames in GS
inpaint with fooocus
'''
import torch
import numpy as np
from ops.llava import Llava
from ops.gs.basic import Frame
from ops.fooocus import Fooocus

class Inpaint_Tool():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self._load_model()
        
    def _load_model(self):
        self.fooocus = Fooocus()
        self.llava = Llava(device='cpu',llava_ckpt=self.cfg.model.vlm.llava.ckpt)

    def _llava_prompt(self,frame):
        prompt = '<image>\n \
                USER: Detaily imagine and describe the scene this image taken from? \
                \n ASSISTANT: This image is taken from a scene of ' 
        return prompt   

    def __call__(self, frame:Frame, outpaint_selections=[], outpaint_extend_times=0.0):
        '''
        Must be Frame type
        '''
        # conduct reconstuction
        # ----------------------- LLaVA -----------------------
        if frame.prompt is None:
            print('Inpaint-Caption[1/3] Move llava.model to GPU...')
            self.llava.model.to('cuda')
            print('Inpaint-Caption[2/3] Llava inpainting instruction:')
            query  = self._llava_prompt(frame)
            prompt = self.llava(frame.rgb,query)
            split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') + len(f'ASSISTANT: This image is taken from a scene of ')
            prompt = prompt[split:]
            print(prompt) 
            print('Inpaint-Caption[3/3] Move llava.model to CPU...')
            self.llava.model.to('cpu')
            torch.cuda.empty_cache()
            frame.prompt = prompt
        else:
            prompt = frame.prompt
            print(f'Using pre-generated prompt: {prompt}')
        # --------------------- Fooocus ----------------------
        print('Inpaint-Fooocus[1/2] Fooocus inpainting...')
        image = frame.rgb
        mask = np.zeros_like(image,bool) if len(outpaint_selections)>0 else frame.inpaint
        fooocus_result = self.fooocus(image_number=1,
                            prompt=prompt + ' 8K, no large circles, no cameras, no fisheye.',
                            negative_prompt='Any fisheye, any large circles, any blur, unrealism.',
                            outpaint_selections=outpaint_selections,
                            outpaint_extend_times=outpaint_extend_times,
                            origin_image=image,
                            mask_image=mask,)[0]
        torch.cuda.empty_cache()
        
        # reset the frame for outpainting
        if len(outpaint_selections) > 0.:
            assert len(outpaint_selections) == 4
            small_H, small_W = frame.rgb.shape[0:2]
            large_H, large_W = fooocus_result.shape[0:2]
            if frame.intrinsic is not None:
                # NO CHANGE TO FOCAL
                frame.intrinsic[0,-1] = large_W//2 
                frame.intrinsic[1,-1] = large_H//2 
            # begin sample pixel
            frame.H = large_H
            frame.W = large_W
            begin_H = (large_H-small_H)//2
            begin_W = (large_W-small_W)//2
            inpaint = np.ones_like(fooocus_result[...,0])
            inpaint[begin_H:(begin_H+small_H),begin_W:(begin_W+small_W)] *= 0.
            frame.inpaint = inpaint > 0.5
        frame.rgb = fooocus_result
        
        print('Inpaint-Fooocus[2/2] Assign Frame...')
        return frame
    