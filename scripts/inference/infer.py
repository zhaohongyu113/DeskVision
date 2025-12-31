import os, sys
import json
from pathlib import Path
import re
import argparse
import numpy as np

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image, ImageDraw, ImageFont
import copy
import torch
import warnings

warnings.filterwarnings("ignore")


def _load_infer_config() -> dict:
    """Load optional inference config.

    Search order (first match wins):
    1) scripts/inference/config.json (next to this file)
    """
    config_path = Path(__file__).resolve().with_name("config.json")
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_pretrained_models_path() -> str:
    """Determine pretrained model directory.

    Priority:
    1) env var DESKVISION_PRETRAINED_MODELS
    2) scripts/inference/config.json: {"pretrained_models": "..."}
    3) historical default ../../pretrained_models
    """
    env_value = os.environ.get("DESKVISION_PRETRAINED_MODELS")
    if env_value:
        return env_value
    cfg = _load_infer_config()
    cfg_value = cfg.get("pretrained_models") if isinstance(cfg, dict) else None
    if isinstance(cfg_value, str) and cfg_value.strip():
        return cfg_value
    return "../../pretrained_models"


def denormalize(bbox, image_size):
    bbox = [
        int(bbox[0] / 999 * image_size[0]),
        int(bbox[1] / 999 * image_size[1]),
        int(bbox[2] / 999 * image_size[0]),
        int(bbox[3] / 999 * image_size[1])
    ]
    return bbox


def parse_bbox(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    pattern1 = r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]'
    pattern2 = r'\<bbox>\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\</bbox>'
    pattern3 = r'\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)'
    # Use re.search to find the first match of the pattern in the input string
    match1 = re.search(pattern1, input_str)
    match2 = re.search(pattern2, input_str)
    match3 = re.search(pattern3, input_str)
    res = None
    # If a match is found, convert the captured groups into a list of floats
    if match1:
        res = [float(match1.group(i)) for i in range(1, 5)]
    elif match2:
        res = [float(match2.group(i)) for i in range(1, 5)]
    elif match3:
        res = [float(match3.group(i)) for i in range(1, 5)]
    
    # If the input does not contain the pattern, return the null float sequence
    return res


def draw_bbox_with_text(image_path, bbox, text, output_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    bbox = denormalize(bbox, image.size)
    draw = ImageDraw.Draw(image)

    # 绘制边界框
    draw.rectangle(bbox, outline="blue", width=3)

    # 标注文本
    # font = ImageFont.load_default()  # 使用默认字体
    font = ImageFont.truetype("default.ttf", 15)
    if bbox[1]-30<20:  
        text_position = (bbox[0], bbox[1] + 30)  # 将文本放在bbox的上方
    else:
        text_position = (bbox[0], bbox[1] - 30)
    draw.text(text_position, text, fill="blue", font=font)

    # 保存处理后的图片
    image.save(output_path)
    # print(f"Image saved with bbox and text at {output_path}")
    return output_path


def draw_bbox_with_text_list(image_path, bboxs, texts, output_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    image_size = image.size
    font = ImageFont.truetype("default.ttf", 15)
    for bbox, text in zip(bboxs, texts):
        try:
            bbox = denormalize(bbox, image_size)
            # 绘制边界框
            draw.rectangle(bbox, outline="blue", width=3)

            # 标注文本
            # font = ImageFont.load_default()  # 使用默认字体
            if bbox[1]-30<20:  
                text_position = (bbox[0], bbox[1] + 30)  # 将文本放在bbox的上方
            else:
                text_position = (bbox[0], bbox[1] - 30)
            draw.text(text_position, text, fill="red", font=font)
        except:
            print("text: {}, bbox: {} is failed".format(text, bbox))
            continue
    # 保存处理后的图片
    image.save(output_path)
    # print(f"Image saved with bbox and text at {output_path}")


def infer_model(image_tensor, text_input, image_sizes):
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + text_input
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # print("instruction task; input: {}, output: {}".format(text_input, text_outputs[0]))
    return text_outputs


def ocr(image_path, text_input=""):
    bbox = parse_bbox(text_input)
    img_name = os.path.basename(image_path)
    image = Image.open(image_path)
    image_sizes = [image.size]
    w, h = image.size
    if bbox is not None and bbox[0] >= 1:
        bbox[0] = str(int(bbox[0] / w * 999))
        bbox[1] = str(int(bbox[1] / h * 999))
        bbox[2] = str(int(bbox[2] / w * 999))
        bbox[3] = str(int(bbox[3] / h * 999))
    else:
        print('ocr task: input bbox is invaild! Currently supported "(x1, y1, x2, y2)", "[x1, y1, x2, y2]", and it must be absolute coordinates')
        return None, None, None
    bbox_str = ",".join(bbox)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    text_input_revised = '识别边界框中的文本<bbox>{}</bbox>'.format(bbox_str)
    text_outputs = infer_model(image_tensor, text_input_revised, image_sizes)
    answer = text_outputs[0].replace("<ocr>", "").replace("</ocr>", "")
    output_path = os.path.join(save_root, "ocr_" + img_name)
    bbox = [int(item) for item in bbox]
    output_path = draw_bbox_with_text(image_path, bbox, answer, output_path)
    print("ocr task; input: {}, output: {}".format(text_input, text_outputs[0]))
    return output_path, ""


def grounding(image_path, text_input=""):
    img_name = os.path.basename(image_path)
    image = Image.open(image_path)
    image_sizes = [image.size]
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    text_input_revised = "输出这个文本的边界框 <ocr> {} </ocr>".format(text_input)
    text_outputs = infer_model(image_tensor, text_input_revised, image_sizes)
    answer = text_outputs[0].replace("<bbox>", "").replace("</bbox>", "").split(",")
    bbox = [float(item) for item in answer]
    bbox_abs = denormalize(bbox, image.size)
    print("grounding task; input: {}, output: {}".format(text_input, bbox_abs))
    output_path = os.path.join(save_root, "grounding_" + img_name)
    output_path = draw_bbox_with_text(image_path, bbox, text_input, output_path)
    return output_path, ""


def instruction(image_path, text_input=""):
    img_name = os.path.basename(image_path)
    image = Image.open(image_path)
    image_sizes = [image.size]
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    text_outputs = infer_model(image_tensor, text_input, image_sizes)
    try:
        answer = text_outputs[0].split("<bbox>")[-1].split("</bbox>")[0]
        bbox = [float(item) for item in answer.split(",")]
        bbox_abs = denormalize(bbox, image.size)
        bbox_abs = [str(cords) for cords in bbox_abs]
        # 将相对坐标替换为绝对坐标输出
        bbox_abs = ",".join(bbox_abs)
        text_outputs[0] = text_outputs[0].replace(answer, bbox_abs)
        output_path = os.path.join(save_root, "ins_" + img_name)
        output_path = draw_bbox_with_text(image_path, bbox, text_outputs[0], output_path)
    except:
        output_path = image_path
    print("instruction task; input: {}, output: {}".format(text_input, text_outputs[0]))
    return output_path, text_outputs[0]


def run_task(task="grounding", text_input="", image_path=""):
    if text_input == "" or image_path == "":
        print("text input or image input cannot be None!")
        return None, None
    if task == "ocr":
        output_img, output_text = ocr(image_path, text_input)
    elif task == "grounding":
        output_img, output_text = grounding(image_path, text_input)
    elif task == "instruction":
        output_img, output_text = instruction(image_path, text_input)
    else:
        print('{} task is not supported! Currently supported "ocr","grounding" and "instruction"')
        return None, None
    return output_img, output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GUI understanding inference scripts')

    parser.add_argument("--task", help='Task type, currently supported "ocr","grounding","instruction"', default='grounding', type=str)

    parser.add_argument('--input_text', help='Input text. If it is "ocr", enter the absolute coordinates "[x1,y1,x2,y2]" of the area to be identified; "grounding" means enter the content to be located; others are instructions;', default='', type=str)
    parser.add_argument('--input_image', help='Input image path to be understood by the model', required=True, type=str)
    parser.add_argument('--pretrained_models', help='the path of pretrained models', default=_default_pretrained_models_path(), type=str)

    args = parser.parse_args()
    pretrained = args.pretrained_models
    save_root = "./visual_results"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
        # "attn_implementation": "sdpa",
        "attn_implementation": None,
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    res_img, res_ins = run_task(args.task, args.input_text, args.input_image)
    print("instructions to execute: {}".format(res_ins))
    print("result image: {}".format(res_img))
