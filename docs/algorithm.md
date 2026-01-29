# 算法/模型如何从掌纹中区分三条主线

本文档结合当前项目实现（`code/read_palm.py` / `code/classification.py` / `code/detection.py`）说明三条主线（心线/智慧线/生命线）从掌纹中区分的具体流程与判定逻辑。

## 1. 总体思路

本项目采用“**掌纹分割 → 骨架化 → 图结构线段提取 → 特征匹配选线**”的流程，不直接做三类分割，而是先提取所有可能线段，再通过特征空间匹配筛出三条主线。

## 2. 关键步骤

### 2.1 掌纹分割（线条检测）

- **输入**：经过校正的掌心图像（`rectification.warp_with_matrix`）。
- **模型**：U-Net（`model.UNet`），输出掌纹区域的二值 mask。  
- **实现位置**：
  - 推理入口：`code/read_palm.py` 中 `detect(...)` 的调用；
  - 推理函数：`code/detection.py` 中 `detect(net, jpeg_dir, output_dir, resize_value, device=...)`。
- **关键设定**：输入统一到 `256x256`，阈值 `0.03` 将模型输出转为二值线条。

**核心代码片段（与实现一致）**：

```python
# code/detection.py

def detect(net, jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
    pil_img = Image.open(jpeg_dir)
    img = np.asarray(pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)) / 255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
    pred = net(img).squeeze(0)
    pred = torch.Tensor(
        np.apply_along_axis(lambda x: [1,1,1] if x > 0.03 else [0,0,0], 0, pred.cpu().detach())
    )
    Image.fromarray((pred.permute((1,2,0)).numpy() * 255).astype(np.uint8)).save(output_dir)
```

**说明**：
- `Image.NEAREST` 保留线条的细节边缘，避免插值导致的模糊；
- `pred` 是模型的概率输出，使用固定阈值转成二值掩码；
- 输出是 3 通道的“线条图”，后续用于 skeletonize。

### 2.2 骨架化（细线化）

- 对分割后的掌纹 mask 进行 skeletonize，得到单像素宽的掌纹骨架。
- 目的：将粗线条转为连通曲线，便于后续图结构建模。
- **实现位置**：`code/classification.py` 中 `classify` 里对分割图直接骨架化。

**核心代码片段（与实现一致）**：

```python
# code/classification.py

palmline_img = cv2.imread(path_to_palmline_image)
# kernel = np.ones((3, 3), np.uint8)
# dilated = cv2.dilate(palmline_img, kernel, iterations=3)
# eroded = cv2.erode(dilated, kernel, iterations=3)
skel_img = cv2.cvtColor(skeletonize(palmline_img), cv2.COLOR_BGR2GRAY)
```

**说明**：
- 这里直接对分割结果做 `skeletonize`，获得单像素宽骨架；
- 形态学操作保留为可选项（注释掉的膨胀/腐蚀）。

### 2.3 图结构线段提取

- 将骨架图视为图结构：
  - **节点**：端点（度=1）与交叉点（度≥3）。
  - **边**：节点之间的连续像素序列。
- 通过图回溯（backtracking）枚举所有可能的线段路径。
- 过滤过短或方向异常的线段（长度 < 10 或方向反转）。
- **实现位置**：`code/classification.py` 中 `group` / `backtrack`。

**核心代码片段（与实现一致）**：

```python
# code/classification.py (节选)

def group(img):
    count = np.zeros(img.shape)
    nodes = []

    for j in range(1, img.shape[0] - 1):
        for i in range(1, img.shape[1] - 1):
            if img[j, i] == 0:
                continue
            count[j, i] = np.count_nonzero(img[j-1:j+2, i-1:i+2]) - 1
            if count[j, i] == 1 or count[j, i] >= 3:
                nodes.append((j, i))

    # 构图并回溯，得到所有候选线段
    lines_node = []
    visited_node, finished_node = {}, {}
    for node in nodes:
        visited_node[node] = False
        finished_node[node] = False
    for node in nodes:
        if not finished_node[node]:
            temp = [node]
            visited_node[node] = True
            finished_node[node] = True
            backtrack(lines_node, temp, graph, visited_node, finished_node, node)
```

**说明**：
- 端点与交叉点由 3x3 邻域像素计数得到；
- 图回溯产生的线段再根据长度与方向一致性做过滤。

### 2.4 三条主线选择（特征匹配）

- 对每条候选线段提取特征：
  - 线段的最小/最大边界（min/max x,y）
  - 线段方向的分段均值（10 个区间）
  - 拼接成固定长度的向量（24 维）。
- 与预训练的 3 个 K-means 聚类中心做 L2 距离匹配：
  - 每个中心对应一条“典型主线形态”。
  - 选择距离最近的 3 条候选线段作为主线输出。
- **实现位置**：`code/classification.py` 中 `extract_feature` / `classify_lines` / `get_cluster_centers`。

**核心代码片段（与实现一致）**：

```python
# code/classification.py (节选)

def extract_feature(line, image_height, image_width):
    image_size = np.array([image_height, image_width], dtype=np.float32)
    feature = np.append(
        np.min(line, axis=0)[:2] / image_size,
        np.max(line, axis=0)[:2] / image_size,
    )
    feature *= 10
    N = 10
    step = len(line) // N
    for i in range(N):
        l = line[i*step:(i+1)*step]
        feature = np.append(feature, np.mean(l, axis=0)[2:])
    return feature


def classify_lines(centers, lines, image_height, image_width):
    classified_lines = [None, None, None]
    line_idx = [None, None, None]
    nearest = [1e9, 1e9, 1e9]

    feature_list = np.empty((0,24))
    for line in lines:
        feature = extract_feature(line, image_height, image_width)
        feature_list = np.vstack((feature_list, feature))

    for i in range(3):
        center = centers[i]
        for j in range(len(lines)):
            if j in line_idx[:i]:
                continue
            dist = np.linalg.norm(feature_list[j] - center)
            if dist < nearest[i]:
                nearest[i] = dist
                classified_lines[i] = lines[j]
                line_idx[i] = j
    return classified_lines
```

**说明**：
- 线段位置（min/max）编码了“位于掌心上/中/下”的信息；
- 方向均值编码了“走向、弯曲程度”的信息；
- K-means 中心在 `get_cluster_centers` 中给出（当前项目使用固定中心）。

## 3. 三条主线的判定依据

该项目的“心线/智慧线/生命线”并非通过语义标签直接预测，而是通过**形态位置分布 + 方向特征**间接区分：

- **心线（Heart line）**：
  - 通常位于掌心上部（靠近指根），走向较水平。
- **智慧线（Head line）**：
  - 位于掌心中部，走向横向或微向下倾斜。
- **生命线（Life line）**：
  - 位于掌心下部，通常为弧形围绕大拇指根部。

在实现中，三条线的形态与位置被编码进聚类中心，最终通过“特征相似度”完成区分。

## 4. 与直接分类的差异

- **当前方案**：先找所有线段，再匹配出三条主线（特征聚类）。
- **直接方案**：在分割阶段直接输出三类（心/智/生）。
- **原因**：
  - 多类分割需要大量精细标注数据；
  - 当前数据与时间成本更适合“后处理分类”的策略。

## 5. 局限性与改进方向

- **局限**：
  - 如果骨架断裂或噪声过多，会产生错误候选线。
  - K-means 中心来自固定样本分布，跨人群/光照可能偏移。

- **改进方向**：
  1. 使用多类分割模型直接预测三条线；
  2. 用更多样本重新训练聚类中心或改用监督分类；
  3. 加入拓扑约束（生命线应环绕拇指根部等规则）。

## 6. 小结

该项目通过“分割 + 骨架 + 图结构 + 特征聚类”的方式，将掌纹中的所有线段归约为三条主线。核心在于：**先完整提取线条结构，再用形态特征与模板中心匹配完成主线识别**。
