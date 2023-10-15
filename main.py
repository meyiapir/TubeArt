import glob
import os
from io import BytesIO

import cv2
import gradio as gr
import torch
from PIL import Image
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPVisionModelWithProjection

print("Loading models...")

DEVICE = torch.device('cuda:0')

model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor


def caption_image(image_file, prompt):
    if image_file:
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        image_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
    else:
        image_tensor = None
        image_tokens = ""

    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    inp = f"{roles[0]}: {prompt}"
    inp = image_tokens + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    kwargs = {'do_sample': True, 'temperature': 0.3, 'max_new_tokens': 128, 'use_cache': True,
              'stopping_criteria': [stopping_criteria]}
    if image_tensor is not None:
        kwargs['images'] = image_tensor

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **kwargs)

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]

    return image if image_file else None, output


image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    subfolder='image_encoder'
).half().to(DEVICE)

unet = UNet2DConditionModel.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    subfolder='unet'
).half().to(DEVICE)

prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    image_encoder=image_encoder,
    torch_dtype=torch.float16
).to(DEVICE)

decoder = KandinskyV22Pipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    unet=unet,
    torch_dtype=torch.float16
).to(DEVICE)


def clear_frames(folder_path):
    files = glob.glob(f"{folder_path}/*.jpg")
    for f in files:
        os.remove(f)


def process_video_and_generate_cover(video_path: gr.inputs.Video):
    print("Processing video...")
    output_folder = "key_frames"

    # Список для хранения путей к ключевым кадрам
    key_frames_paths = []

    # Создаем папку, если не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)

    # Получаем кол-во кадров
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Рассчитываем шаг
    step = total_frames // 10

    for i in range(10):
        # Перемещаемся к нужному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)

        ret, frame = cap.read()

        if ret:
            output_path = os.path.join(output_folder, f"key_frame_{i}.jpg")
            cv2.imwrite(output_path, frame)
            key_frames_paths.append(output_path)

    # Освобождаем ресурсы
    cap.release()

    # Шаг 2: Описания через LLAVA
    descriptions = {}  # Словарь для хранения описаний
    for frame_path in key_frames_paths:
        _, desc = caption_image(frame_path, 'Describe the image.')
        trimmed_desc = " ".join(desc.split())
        descriptions[frame_path] = trimmed_desc

    # Шаг 3: Выбор финального промпта через LLM
    prompt_input = "Your task is to create a list of key phrases (separated by commas) based on the descriptions of the video frames that will help the neural network artist to draw a cover for this video: " + ', '.join(
        list(descriptions.values()))
    _, final_prompt = caption_image(None,
                                    prompt_input)  # Подразумевается, что функция caption_image может работать без изображения

    negative_prior_prompt = 'lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

    if not final_prompt:
        print("Error! Final prompt is empty.")
        exit(1)
    print("Final prompt:", final_prompt)

    # Шаг 4: Генерация обложки через Kandinsky
    img_emb = prior(
        prompt="video cover, quality, realistic" + final_prompt,
        num_inference_steps=75,
        num_images_per_prompt=1
    )

    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=75,
        num_images_per_prompt=1
    )

    images = decoder(
        image_embeds=img_emb.image_embeds,
        negative_image_embeds=negative_emb.image_embeds,
        num_inference_steps=75,
        height=512,
        width=720)

    cover_image = images.images[0]

    # Шаг 5: Удаление фреймов
    clear_frames(output_folder)
    return cover_image


video_input_component = gr.Video(file=True)
image_output_component = gr.Image(type="pil")

interface = gr.Interface(
    fn=process_video_and_generate_cover,
    inputs=video_input_component,
    outputs=image_output_component,
    enable_queue=True,
    title="TubeArt",
    debug=True
)

if __name__ == "__main__":
    interface.queue().launch(share=True)
