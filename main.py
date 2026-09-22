import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

# Importaciones de IDM-VTON
from utils_mask import get_mask_location
import apply_net

def run_idm_vton(
    human_img_path: str,
    garment_img_path: str,
    garment_description: str,
    category: str = "upper_body",  # Usar "upper_body" para TOP o "lower_body" para BOTTOM
    is_checked_crop: bool = False,
    denoise_steps: int = 30,
    seed: int = 42,
    device: str = "cuda"
):
    """
    Función de inferencia para IDM-VTON soportando TOP y BOTTOM.
    
    :param human_img_path: Ruta a la imagen de la persona.
    :param garment_img_path: Ruta a la imagen de la prenda.
    :param garment_description: Texto descriptivo de la prenda.
    :param category: "upper_body" (para TOP) o "lower_body" (para BOTTOM).
    """
    
    # 1. Cargar y preparar imágenes
    human_img_orig = Image.open(human_img_path).convert("RGB")
    garm_img = Image.open(garment_img_path).convert("RGB").resize((768, 1024))
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    # 2. Generación automática de máscara según la CATEGORÍA (TOP o BOTTOM)
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    
    # Aquí es donde se especifica si es "upper_body" (TOP) o "lower_body" (BOTTOM)
    mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 3. Mapeo de postura DensePose
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        'show', 
        './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        './ckpt/densepose/model_final_162be9.pkl', 
        'dp_segm', 
        '-v', 
        '--opts', 
        'MODEL.DEVICE', 
        device
    ))
    pose_img = args.func(args, human_img_arg)    
    pose_img = pose_img[:, :, ::-1]    
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # 4. Inferencia con el Pipeline de Diffusion
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_description
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
                                
            prompt_c = "a photo of " + garment_description
            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

            pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
            
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img_tensor.to(device, torch.float16),
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor.to(device, torch.float16),
                mask_image=mask,
                image=human_img, 
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig
    else:
        return images[0]


# ======================================================
# EJEMPLOS DE INVOCACIÓN EN TU REPOSITORIO
# ======================================================

# Para probar una prenda superior (TOP)
resultado_top = run_idm_vton(
    human_img_path="images/persona.jpg",
    garment_img_path="images/remera.jpg",
    garment_description="Short sleeve black t-shirt",
    category="upper_body"
)
resultado_top.save("output_top.png")

# Para probar una prenda inferior (BOTTOM)
resultado_bottom = run_idm_vton(
    human_img_path="images/persona.jpg",
    garment_img_path="images/pantalon.jpg",
    garment_description="Blue denim pants",
    category="lower_body"
)
resultado_bottom.save("output_bottom.png")
