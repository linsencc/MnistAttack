## 手写数字识别

### 输入参数说明
```shell
optional arguments:

  --image(str) 图片的路径 

  --method(str) {identify,fgsm,ifgsm,mifgsm,deefool} 识别或攻击方法
  
  --eps(float) 生成对抗样本时其扰动大小限制

  --interation(int) 生成对抗样本时其迭代次数

  --alpha(str) 生成对抗样本时其迭代步长

  --decay(str) MI-FGSM攻击中的decay系数
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
python3 handler.py --image /Users/bytedance/Desktop/9.ca9a22b3.jpg --method identify # 要且只要命令行中使用到的参数

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


