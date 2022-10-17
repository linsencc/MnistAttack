# coding:utf-8
import argparse
import os
import traceback
import numpy as np
import cv2
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models import LeNet5
import attack


parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('--image', type=str, help='图片的路径', required=True)
parser.add_argument('--method', type=str, help='识别或攻击方法', choices=['identify', 'fgsm', 'ifgsm', 'mifgsm', 'deefool'], required=True)
parser.add_argument('--eps',  type=float, help='生成对抗样本时其扰动大小限制')
parser.add_argument('--interation', type=int, help='生成对抗样本时其迭代次数')
parser.add_argument('--alpha', type=float, help='生成对抗样本时其迭代步长')
parser.add_argument('--decay', type=float, help='MI-FGSM攻击中的decay系数')


# 加载模型
Model = LeNet5()
LoadData = torch.load('./save/custom-train.pt', map_location=torch.device('cpu'))
Model.load_state_dict(LoadData['model_state_dict'])
Model.eval()


def get_tensor_image(img_file: str):
    pil_img = Image.open(img_file).convert('L')
    pil_img = pil_img.resize((24, 24))
    pil_img = ImageOps.invert(pil_img)
    tensor_image = transforms.ToTensor()(pil_img)
    tensor_image = tensor_image.reshape(1, 1, 24, 24)
    return tensor_image


def detect(tensor_image: torch.tensor):
    output = Model(tensor_image)
    value, predict = torch.max(output, 1)
    predict_distribution = F.softmax(output, dim=1)
    return predict.tolist(), predict_distribution.tolist()


def get_attacker(param: dict):
    ack_method = str(param.get('method', ''))
    attacker = None

    if ack_method == 'fgsm':
        eps = float(param.get('eps'))
        attacker = attack.FGSM(Model, eps)
    if ack_method == 'ifgsm':
        eps = float(param.get('eps'))
        iteration = float(param.get('iteration'))
        alpha = float(param.get('alpha'))
        attacker = attack.IFGSM(Model, eps, alpha, iteration)
    if ack_method == 'mifgsm':
        eps = float(param.get('eps'))
        iteration = float(param.get('iteration'))
        alpha = float(param.get('alpha'))
        decay = float(param.get('decay'))
        attacker = attack.MIFGSM(Model, eps, alpha, iteration, decay)
    if ack_method == 'deepfool':
        eps = float(param.get('eps'))
        iteration = float(param.get('iteration'))
        attacker = attack.DeepFool(eps, iteration)
    return attacker


def identify(param):
    image_path = param.get('image', '')
    img_tensor = get_tensor_image(image_path)
    predict, predict_distribution = detect(img_tensor)
    result = {"predict": predict, "distribution": predict_distribution[0]}
    return result


def mnist_attack(param):
    image_path = param.get('image', '')
    dir_path, image_name = os.path.split(image_path)
    img_tensor = get_tensor_image(image_path)

    attacker = get_attacker(param)
    img_adv = attacker(img_tensor)
    predict, predict_distribution = detect(img_adv)

    # 对抗样本图片，PIL库直接转化会有损失，这里使用了cv2
    np_adv_image = np.array(img_adv * 255, np.int16)
    np_adv_image = np_adv_image.reshape((24, 24))
    # 样本原图
    np_ori_image = np.array(img_tensor[0][0] * 255, dtype=np.int16)
    np_ori_image = np_ori_image.reshape((24, 24))
    # 获取对抗样本和原图的差值（生成的噪声），并进行反转（白底个人觉得更直观）
    np_noise_image = 255 - (np_adv_image - np_ori_image)
    np_adv_image = 255 - np_adv_image
    # cv2写入图像
    adv_image_path = os.path.join(dir_path, 'adv_' + image_name)
    cv2_adv_resp = cv2.imwrite(adv_image_path, np_adv_image)
    noise_image_path = os.path.join(dir_path, 'noise_' + image_name)
    cv2_noise_resp = cv2.imwrite(noise_image_path, np_noise_image)

    return {"predict": predict,
            "distribution": predict_distribution[0],
            "adv_image_path": adv_image_path if cv2_adv_resp else "",
            "noise_image_path": noise_image_path if cv2_noise_resp else ""}


def process():
    param = vars(parser.parse_args())
    method = param.get('method', 'identify')
    try:
        resp = {'code': 0, 'msg': 'success'}
        if method == 'identify':
            resp['data'] = identify(param)
        else:
            resp['data'] = mnist_attack(param)
    except Exception as e:
        ex_msg = traceback.format_exc()
        resp = {'code': 1, 'msg': ex_msg, 'data': {}}
    return resp


if __name__ == '__main__':
    data = process()
    print(data)
