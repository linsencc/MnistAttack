# coding:utf-8

from flask import Flask, request
from flask import make_response
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models import LeNet5
import attack


app = Flask(__name__)
Model = LeNet5()
LoadData = torch.load('./save/custom-train.pt', map_location=torch.device('cpu'))
Model.load_state_dict(LoadData['model_state_dict'])
Model.eval()


def get_tensor_image(img_file):
    pil_img = Image.open(img_file).convert('L')
    pil_img = pil_img.resize((24, 24))
    pil_img = ImageOps.invert(pil_img)
    tensor_image = transforms.ToTensor()(pil_img)
    tensor_image = tensor_image.reshape(1, 1, 24, 24)
    return tensor_image


def detect(tensor_image):
    output = Model(tensor_image)
    value, predict = torch.max(output, 1)
    predict_distribution = F.softmax(output, dim=1)
    return predict.tolist(), predict_distribution.tolist()


def get_attacker(param):
    ack_method = str(param.get('method', ''))
    eps = float(param.get('eps', 0))
    iteration = float(param.get('iteration', 0))
    alpha = float(param.get('alpha', 0))
    decay = float(param.get('decay', 0))

    attacker = None
    if ack_method == 'fgsm':
        attacker = attack.FGSM(Model, eps)
    if ack_method == 'ifgsm':
        attacker = attack.IFGSM(Model, eps, alpha, iteration)
    if ack_method == 'mifgsm':
        attacker = attack.MIFGSM(Model, eps, alpha, iteration, decay)
    if ack_method == 'deepfool':
        attacker = attack.DeepFool(eps, iteration)

    return attacker


@app.route('/mnistapplication', methods=['POST'])
def mnistapplication():
    img_file = request.files['img']
    img_tensor = get_tensor_image(img_file)
    predict, predict_distribution = detect(img_tensor)
    result = make_response({"preValue": str(predict)}) if predict else make_response({"preValue": "1001"})
    return result


@app.route('/mnistattack', methods=['POST'])
def mnistattack():
    img_file = request.files['img']
    param = request.values

    img_tensor = get_tensor_image(img_file)
    attacker = get_attacker(param)
    img_adv = attacker(img_tensor)

    pred_ori = detect(img_tensor)
    pred_adv = detect(img_adv)

    return make_response({pred_ori, pred_adv})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
