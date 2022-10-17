## 手写数字识别

### 输入参数说明
```shell
optional arguments:

  --image(str) 图片的路径 

  --method(str) {identify,fgsm,ifgsm,mifgsm,deefool} 识别或攻击方法
  
  --eps(float) 生成对抗样本时其扰动大小限制，范围 0-40

  --interation(int) 生成对抗样本时其迭代次数，范围 0-40

  --alpha(str) 生成对抗样本时其迭代步长，范围 0-10

  --decay(str) MI-FGSM攻击中的decay系数，范围 0-1 
```

### 输出参数说明
```shell
# 执行成功：
# 返回code字段固定为0, msg字段固定为success，data字段为具体信息
{'code': 0, 'msg': 'success', 'data': {...}}

# 执行失败
# 返回code字段固定为1, msg字段为栈错误信息，data字段为空
{'code': 1, 'msg': 'err msg', 'data': {}}
```

### 例子
#### 1. 识别图片
```shell
# 要且只要命令行中使用到的参数
python3 handler.py --image 5.jpg --method identify 

# 返回值
{ 
  'code': 0, 
  'msg': 'success', 
  'data': {
    'predict': [9],  # 预测值
    'distribution': [ 0.0006740741664543748, # 预测分布
                      0.006121807731688023, 
                      0.007856828160583973, 
                      0.008956621401011944, 
                      0.114366315305233, 
                      0.0011688891099765897, 
                      4.784505654242821e-05, 
                      0.3034481108188629, 
                      0.002582007320597768, 
                      0.5547774434089661]
  }
}
```

#### 2. fgsm 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method ifgsm --eps 20 --iteration 10 --alpha 2

# 返回值（所有攻击的返回值格式一致）
{ 
  'code': 0, 
  'msg': 'success', 
  'data': {
    'predict': [9],  # 预测值
    'distribution': [ 0.0006740741664543748, # 预测分布
                      0.006121807731688023, 
                      0.007856828160583973, 
                      0.008956621401011944, 
                      0.114366315305233, 
                      0.0011688891099765897, 
                      4.784505654242821e-05, 
                      0.3034481108188629, 
                      0.002582007320597768, 
                      0.5547774434089661]
    'adv_image_path': 'adv_5.jpg',          # 生成对抗样本图片所保存路径，保存路径与输入图片一致
    'noise_image_path': 'noise_5.jpg'       # 生成对抗噪声图片所保存路径，保存路径与输入图片一致
  }
}
```

#### 2. ifgsm 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method ifgsm --eps 15 --iteration 10 --alpha 2

# 返回值（所有攻击的返回值格式与fgsm攻击一致）
```

#### 3. mifgsm 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method mifgsm --eps 15 --iteration 10 --alpha 2 --decay 0.8

# 返回值（所有攻击的返回值格式与fgsm攻击一致）
```

#### 4. deepfool 攻击
```shell script
# 要且只要命令行中使用到的参数
python handler.py --image 5.jpg --method deepfool --eps 15 --iteration 10

# 返回值（所有攻击的返回值格式与fgsm攻击一致）
```