# 测试样例评估报告

## 1. 样例信息

- 样例编号：S-001

- 文件名：`input/hand_01.jpg`

  ![image-20260130110320004](E:\model-1.26\Internship\CarryCode\palmistry\docs\test.assets\image-20260130110320004.png)

- 分辨率：1920x1080

- 拍摄条件：自然光，背景偏浅色，手掌正对镜头

- 期望结果：
  - 能识别手掌 ROI
  - 输出心线、智慧线、生命线三条主线和虎口、掌心、掌根三个关键点
  - 叠加显示清晰，不遮挡关键掌纹

## 2. 运行配置

- 运行入口：h5前端导入图片
- 模型：`checkpoint/checkpoint_aug_epoch70.pth`
- 关键参数：
  - 预处理：HSV 手部阈值抠图
  - 校正：MediaPipe Hands + Homography
  - 分割：U-Net，输入尺寸 256x256

## 3. 结果记录

- 是否成功识别：是
- 运行耗时：**1282 ms**（CPU）
- 感情线置信度**69%**

  智慧线置信度**71%**

  生命线置信度**73%**
- 输出文件：
  - `results/warped_palm.jpg`
  - `results/palm_lines.png`
  - `results/result.jpg`
  - `results/keypoints.json`
- 叠加效果主观评价：
  - 心线：清晰，走向合理
  - 智慧线：部分断裂，但整体可辨
  - 生命线：起点正确，尾部略短

![image-20260130111435869](E:\model-1.26\Internship\CarryCode\palmistry\docs\test.assets\image-20260130111435869.png)

------



## 1. 样例信息

- 样例编号：S-002

- 文件名：`input/hand_02.jpg`

  ![image-20260130110959527](E:\model-1.26\Internship\CarryCode\palmistry\docs\test.assets\image-20260130110959527.png)

- 分辨率：1920x1080

- 拍摄条件：自然光，背景偏深色，手掌正对镜头，但是**手掌侧着摆放**

- 期望结果：

  - 能识别手掌 ROI
  - 输出心线、智慧线、生命线三条主线和虎口、掌心、掌根三个关键点
  - 叠加显示清晰，不遮挡关键掌纹

## 2. 运行配置

- 运行入口：h5前端导入图片
- 模型：`checkpoint/checkpoint_aug_epoch70.pth`
- 关键参数：
  - 预处理：HSV 手部阈值抠图
  - 校正：MediaPipe Hands + Homography
  - 分割：U-Net，输入尺寸 256x256

## 3. 结果记录

- 是否成功识别：是
- 运行耗时：**5550 ms**（CPU）
- 感情线置信度**70%**

  智慧线置信度**72%**

  生命线置信度**78%**
- 输出文件：
  - `results/warped_palm.jpg`
  - `results/palm_lines.png`
  - `results/result.jpg`
  - `results/keypoints.json`
- 叠加效果主观评价：
  - 心线：清晰，走向合理
  - 智慧线：尾部略短，但整体可辨
  - 生命线：错误地将虎口的褶皱识别为生命线，但整体可辨

![image-20260130111538675](E:\model-1.26\Internship\CarryCode\palmistry\docs\test.assets\image-20260130111538675.png)

------

## 1. 样例信息

- 样例编号：S-002

- 文件名：`input/hand_02.jpg`

  ![image-20260130185616196](E:\model-1.26\Internship\CarryCode\palmistry\docs\test.assets\image-20260130185616196.png)

- 分辨率：1920x1080

- 拍摄条件：自然光，背景偏深色，手掌正对镜头，但是**手掌倒着摆放，并且手指在图片之外**

- 期望结果：

  - 能识别手掌 ROI
  - 输出心线、智慧线、生命线三条主线和虎口、掌心、掌根三个关键点
  - 叠加显示清晰，不遮挡关键掌纹

## 2. 运行配置

- 运行入口：h5前端导入图片
- 模型：`checkpoint/checkpoint_aug_epoch70.pth`
- 关键参数：
  - 预处理：HSV 手部阈值抠图
  - 校正：MediaPipe Hands + Homography
  - 分割：U-Net，输入尺寸 256x256

## 3. 结果记录

- 是否成功识别：是

- 运行耗时：**11710 ms**（CPU）

- 感情线置信度**80%**

  智慧线置信度**76%**

  生命线置信度**75%**

- 输出文件：

  - `results/warped_palm.jpg`
  - `results/palm_lines.png`
  - `results/result.jpg`
  - `results/keypoints.json`

- 叠加效果主观评价：

  - 心线：清晰，走向合理，但起点和尾部略短
  - 智慧线：起点和尾部略短，但整体可辨
  - 生命线：起点和尾部略短，但整体可辨

![image-20260130185731801](E:\model-1.26\Internship\CarryCode\palmistry\docs\test.assets\image-20260130185731801.png)

------



## 误差与原因分析

- 误差类型：智慧线局部断裂
- 可能原因：
  - 原图该区域光照较强，掌纹对比度不足
  - HSV 抠图造成纹理细节损失

## 结论与改进建议

- 结论：该样例可达到“可用级别”输出，满足三线可辨识要求。
- 改进建议：
  1. 引入更鲁棒的背景分割；
  2. 对分割结果做断裂修复。
