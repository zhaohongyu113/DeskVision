import gradio as gr
import datetime, time
import base64
import uuid
import json
from pathlib import Path
import os, sys
import numpy as np
import traceback
import _thread as thread

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image, ImageDraw, ImageFont
import requests
import copy
import torch
import warnings
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

MAX_NEW_TOKENS = int(os.environ.get("DESKVISION_MAX_NEW_TOKENS", "512"))


def _get_default_font(size: int = 15) -> ImageFont.ImageFont:
    font_path = Path(__file__).resolve().with_name("default.ttf")
    try:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    except Exception:
        pass
    return ImageFont.load_default()


def _load_infer_config() -> dict:
    config_path = Path(__file__).resolve().with_name("config.json")
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_pretrained_models_path() -> str:
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


def draw_bbox_with_text(image_path, bbox, text, output_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    bbox = denormalize(bbox, image.size)
    draw = ImageDraw.Draw(image)

    # 绘制边界框
    draw.rectangle(bbox, outline="blue", width=3)

    # 标注文本
    font = _get_default_font(15)
    if bbox[1]-30<20:  
        text_position = (bbox[0], bbox[1] + 30)  # 将文本放在bbox的上方
    else:
        text_position = (bbox[0], bbox[1] - 30)
    draw.text(text_position, text, fill="blue", font=font)

    # 保存处理后的图片
    image.save(output_path)
    print(f"Image saved with bbox and text at {output_path}")
    return output_path


def draw_bbox_with_text_list(image_path, bboxs, texts, output_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    image_size = image.size
    font = _get_default_font(15)
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
    print(f"Image saved with bbox and text at {output_path}")


def main():

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
            max_new_tokens=MAX_NEW_TOKENS,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print("instruction task; input: {}, output: {}".format(text_input, text_outputs[0]))
        return text_outputs
    
    def app_grounding(image_path, text_input=""):
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

    def app_instruction(image_path, text_input=""):
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
        return output_path, text_outputs[0]

    def app(image_path, text_input="", instruction=""):     
        if text_input == "" and instruction != "":
            output_img, output_text = app_instruction(image_path, instruction)
        elif text_input != "":
            output_img, output_text = app_grounding(image_path, text_input)
        else:
            output_img, output_text = image_path, ""
        return output_img, output_text


    demo = gr.Interface(fn=app,
                    inputs=[
                        gr.Image(label="测试图像", type='filepath'),
                        gr.Textbox(label='定位内容', value=""),
                        gr.Textbox(label='单步指令', value="")
                    ],
                    outputs=[gr.Image(label='结果', type='filepath'),
                             gr.Textbox(label='指令解析', value="")
                    ],
                    examples=[
                        ['./test_imgs/00004.png', '科学计算与物理仿真', ''],
                        ['./test_imgs/00060.png', '开始学习', ''],
                        ['./test_imgs/pw_2.png', '个人成长', ''],
                        ['./test_imgs/taisu_10.jpg', '4800万高清四摄', ''],
                        ['./test_imgs/pw_2.png', '', '我想联系会明心理机构'],
                        ['./test_imgs/pw_1.png', '', '我想打开桌面']
                    ],
                    title="""<div style="font-size:80px">GUI图像理解demo</div>""",
                    description="""<center><div style="font-size:25px">
                                <a>GUI图像理解demo</a>
                                </div>
                                </center>
                                该系统有两个功能，分别是Grounding（文本定位）、单步指令定位。
                                <br /><b>1. Grounding（文本定位）</b>
                                <br />介绍：给定图片内出现的文字，定位该区域（返回定位图，蓝色为预测框），暂不支持icon图标理解定位；
                                <br />操作指南：本地上传测试图和传入待定位文本，点击“submit”返回定位图结果。
                                <br /><b>2. 单步指令定位</b>
                                <br />介绍：该功能需用户给个待执行的单步指令，比如"我想理解本网页的xx内容"，返回待执行的操作指令以及坐标位置；
                                <br />操作指南：本地上传测试图和传入待执行的单步指令(但定位内容处需为空！不然还是Grounding定位），点击“submit”之后demo返回预测操作指令及坐标位置。
                                <br />   
                                
                                """,
                    article="""<div style="font-size:20px">如有其它疑问，请联系作者</div>"""

                    )

    demo.launch(server_name='0.0.0.0',
                server_port=9527,
                share=True
                # enable_queue=True
                )


if __name__ == "__main__":
    pretrained = _default_pretrained_models_path()
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
    main()
