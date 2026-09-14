# Deformable-MSCNN-for-hydrate-Morphologies
# 水合物赋存形态识别代码使用说明

版本日期：2026-09-14。

本文介绍水合物赋存形态识别代码的环境配置、开放数据获取、输入格式、模型结构及训练和预测流程。

## 1. 数据说明

### 1.1 ODP Leg 204 开放数据集与论文依据

**本研究使用的 ODP Leg 204 数据属于公开开放的科学钻探数据，读者可以通过官方数据平台获取。** ODP 是 Ocean Drilling Program（大洋钻探计划），Leg 204 指 2002 年开展的第 204 航次。该航次围绕 Hydrate Ridge 天然气水合物开展调查，资料包括测井、岩心及相关地球化学观测；本文选用其中 1244E、1247B、1252A 三个钻孔，并非整个航次的全部数据。

### 1.2 官方获取入口

| 资料 | 获取入口 | 主要用途 |
|---|---|---|
| 1244E 测井数据 | [LDEO：Hole 1244E](https://mlp.ldeo.columbia.edu/data/odp/leg204/1244E/) | 下载该孔测井文件与处理说明 |
| 1247B 测井数据 | [LDEO：Hole 1247B](https://mlp.ldeo.columbia.edu/data/odp/leg204/1247B/) | 下载该孔测井文件与处理说明 |
| 1252A 测井数据 | [LDEO：Hole 1252A](https://mlp.ldeo.columbia.edu/data/odp/leg204/1252A/) | 下载该孔测井文件与处理说明 |
| 航次测井综述 | [Leg 204 Logging Summary](https://mlp.ldeo.columbia.edu/data/odp/odp-log_sum/leg204/204.index.html) | 查询工具、测量井段、质量控制及地质背景 |
| 航次正式报告 | [ODP Leg 204 Initial Reports](https://www-odp.tamu.edu/publications/204_IR/204ir.htm) | 查询站位报告、方法、岩心描述及相关图表 |
| 岩心及其他航次资料 | [ODP Database Services](https://www-odp.tamu.edu/isg/database.html) | 从官方目录进入 Janus、岩心照片等资源；也可使用各孔数据页的 Site core data 链接 |

以上测井页由哥伦比亚大学 Lamont-Doherty Earth Observatory（LDEO）的测井数据库提供。页面按资料类型提供 Download 链接，可选择 Standard Data（处理后的 ASCII）、High Resolution Data、Documents，以及图像、声波波形或 DLIS 等资料。具体可下载类型以各孔页面为准。

### 1.3 从公开数据到代码输入的步骤

1. 打开上表中三个孔的官方测井页面，确认航次为 204、孔号分别为 1244E、1247B、1252A。不要把同站位的其他孔（如 1244D、1247A）直接替代为论文所用孔。
2. 下载 Documents 与 Standard Data；需要更高分辨率或原始波形时，再选择相应数据类型。点击资料类型可查看文件清单，点击 Download 获取该类归档包。保留原始下载文件及来源记录。
3. 先阅读处理说明和字段/单位说明，再选择电阻率、密度、纵波和横波相关曲线。官方文件的字段名称不一定是 `Rt,Den,Vp,Vs`；如果提供的是声波时差而非速度，应依据明确的单位换算，不能仅重命名为速度列。
4. 对目标井段进行质量控制、深度对齐及必要的重采样，将四项属性整理为第 4 节规定的 `Depth,Rt,Den,Vp,Vs` CSV。
5. 结合航次报告、岩心与红外相关资料核对赋存形态解释。
6. 记录下载日期、孔号、原文件名、测井工具/测次、单位、深度基准、选取井段、处理步骤和标签字典，以便追溯。完成后按照第 4–6 节准备文件并运行模型。

数据来源引用建议：Tréhu, A. M., Bohrmann, G., Rack, F. R., Torres, M. E., et al. (2003). *Proceedings of the Ocean Drilling Program, Initial Reports, 204*. 同时注明所用 LDEO 数据页面及访问日期。

## 2. 文件与模型

| 目录 | 模型定义 | 模型类 | 训练与预测入口 |
|---|---|---|---|
| `CNN` | `cnn_model.py` | `HydrateCNN` | `train_predict.py` |
| `ResNet` | `ResNet_model.py` | `HydrateResNet` | `train_predict.py` |
| `MSCNN` | `MSCNN_model.py` | `HydrateCNN` | `train_predict.py` |
| `SE+MSCNN` | `SE_MSCNN_model.py` | `HydrateCNN` | `train_predict.py` |
| `Deformable Attention+MSCNN` | `Deformable_MSCNN_model.py` | `HydrateCNN` | `train_predict.py` |

### 2.1 网络结构

共同输入为 `[B, 4, 1]`，输出为 `[B, 3]` 的 logits，`B` 是批大小。训练使用交叉熵，模型末尾不需要手动添加 Softmax；预测使用最大 logit 的索引。

| 模型 | 当前代码结构 |
|---|---|
| CNN | Conv1d 4→64，核 3；Conv1d 64→128，核 3；各接 BatchNorm 和 LeakyReLU；全局平均池化；Linear 128→3 |
| ResNet | 起始 Conv1d 4→64，核 7、步长 2；BatchNorm、LeakyReLU、最大池化；64 通道残差块 ×2；128 通道残差块 ×2；全局平均池化；Linear 128→3 |
| MSCNN | 三个并行分支，卷积核 3、5、7，各输出 64 通道并接 BatchNorm、LeakyReLU；拼接为 192 通道；全局平均池化；Linear 192→3 |
| SE MSCNN | 在 MSCNN 每个分支激活后添加 SE 通道注意力；压缩比 16，64→4→64，Sigmoid 门控 |
| Deformable MSCNN | 在 MSCNN 每个分支激活后添加 `DeformableAttention1D(64)`；默认 5 个采样点；卷积生成偏移、采样权重和值特征；聚合后拼接与分类 |

Deformable 模块通过 `tanh` 将偏移限制在 [-1,1]，偏移卷积权重和偏置初始为零，采样位置限制在 [0,L−1]。模块采用一维线性插值，并通过卷积和 Softmax 生成采样权重。

## 3. 环境设置与依赖

### 3.1 运行环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows，命令示例使用 PowerShell |
| Python | 参考版本 3.12.12 |
| 深度学习框架 | PyTorch，已安装版本见下表 |
| 默认执行设备 | CPU |

本机解释器路径为 `D:\Anaconda\envs\gpu\python.exe`。下文命令均使用该解释器，并通过 `-X utf8` 启用 UTF-8。脚本默认在 CPU 上运行，GPU 配置见 8.3 节。

### 3.2 直接依赖

| pip 包名 | import 名 | 用途 | 参考版本 |
|---|---|---|---|
| torch | torch | 网络、自动求导、优化、数据加载与权重保存 | 2.9.1+cu130 |
| numpy | numpy | 数组、异常值处理、归一化、类别计数 | 2.3.5 |
| pandas | pandas | 读取输入与输出 CSV | 2.3.3 |
| scikit-learn | sklearn | 数据处理与分类指标计算 | 1.7.1 |
| imbalanced-learn | imblearn | SMOTE 训练集过采样 | 0.14.2 |
| matplotlib | matplotlib | 训练损失与准确率曲线 | 3.10.6 |

`os` 属于 Python 标准库，无需单独安装。表中版本为本机已安装版本，配置环境时请确认各依赖能够正常导入。

## 4. 输入数据规范

### 4.1 文件名与存放位置

训练脚本统一读取项目根目录的 `hydrate_data.npz`，路径根据脚本位置确定，无需向各模型目录复制数据。

该压缩包包含三口井（`1244E`、`1247B`、`1252A`）原始六个 CSV 的完整数值表格及列名。每口井有四个键：`<井号>_data`、`<井号>_data_columns`、`<井号>_labels`、`<井号>_labels_columns`。使用 `np.load(path, allow_pickle=False)` 读取；特征顺序仍为 `Rt,Den,Vp,Vs`，标签仍取标签表第二列，深度取数据表 `Depth` 列。

原始 CSV 保留用于核对；训练不再读取 CSV。下文 CSV 格式说明描述打包前的源数据格式。预测结果仍输出为 CSV。

### 4.2 特征

第一行为表头，必须包含大小写一致的 `Depth,Rt,Den,Vp,Vs`。可以有其他列，但网络只按 `['Rt','Den','Vp','Vs']` 的顺序读取四项属性。CSV 中列的物理排列顺序可以不同。

| 列名 | 含义 | 使用方式与单位约定 |
|---|---|---|
| Depth | 深度 | 仅保留到输出；图 4 使用 m；应记录深度基准 |
| Rt | 电阻率 | 模型第 1 通道；通常使用 Ω·m，代码不转换单位 |
| Den | 密度 | 模型第 2 通道；代码不转换单位 |
| Vp | 纵波速度 | 模型第 3 通道；论文图 4 标注 km/s |
| Vs | 横波速度 | 模型第 4 通道；论文图 4 标注 km/s |

各属性的单位以源数据说明为准，训练和应用时保持一致；密度数据应在准备 CSV 时明确记录单位。

仅展示格式的虚构示例，不能用于论文训练：

```csv
Depth,Rt,Den,Vp,Vs
80.00,1.20,1.70,1.55,0.25
80.02,1.30,1.72,1.57,0.26
80.04,1.45,1.71,1.59,0.27
```

### 4.3 标签

代码执行 `pd.read_csv(label_file, header=None, skiprows=1)`，然后读取第二列并转为整数。因此文件必须有一行表头，至少两列，第二列为标签。第一列推荐放 Depth，当前代码并不读取或校验它。

```csv
Depth,Label
80.00,0
80.02,1
80.04,2
```

标签使用整数 0、1、2，合并后的训练数据应包含全部三个类别。请按实际标签字典记录各编号对应的无水合物、孔隙充填型和裂隙充填型类别，并在训练与预测中保持一致。上例仅展示编码格式。

特征与标签按行对应，不按深度连接。必须保证行数、排序和深度位置完全一致。标签缺表头时第一条数据会被跳过；标签中的小数会被整数转换截断，因此应在输入前严格检查整数性。

### 4.4 数据准备检查

1. 对测井曲线进行独立质量控制，统一单位、深度基准及采样网格，并将标签对齐到相同深度。
2. 保证四个属性均为数值，不存在字符串占位符；Depth 有效且建议单调排列，重复深度需明确处理。
3. 核对每口井的特征/标签行数及逐行深度一致；确认标签字典为 0、1、2。
4. 检查合并训练集类别分布。默认 SMOTE 的 `k_neighbors=5`，需要被过采样类别至少有 6 个样本；全类别缺失无法靠 SMOTE 补齐。参见 [SMOTE 官方文档](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)。
5. 训练集经 SMOTE 后至少应有一个完整批次。`drop_last=True` 会丢弃最后不足 32 个样本的批次。

请在运行前完成测井质量控制。脚本将 NaN 替换为 0、正无穷替换为 1e6、负无穷替换为 -1e6。

## 5. 预处理与训练参数

每口井独立、每项特征独立计算 `min` 和 `range=max-min`，使用 `(x-min)/range` 归一化。常数列将 range 设为 1，因此归一化后为 0。测试井使用自身全部输入特征的最小值和最大值。

| 参数 | 当前值 / 行为 |
|---|---|
| `WELLS` | 1244E、1247B、1252A |
| `TRAIN_WELLS` | 1244E、1247B，在 main 内设置 |
| `TEST_WELL` | 1252A，在 main 内设置 |
| `FEATURES` | Rt、Den、Vp、Vs |
| `NUM_CLASSES` | 3 |
| `NUM_EPOCHS` | 最多 100 个 epoch，非 100 个 batch |
| `BATCH_SIZE` | 32 |
| 优化器 | Adam，lr=0.003，weight_decay=1e-4 |
| 损失 | 带类别权重的 CrossEntropyLoss |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | patience=10，min_delta=0.001，监控测试井损失 |
| SMOTE | 只对合并训练井执行，random_state=42，其余默认 |
| DataLoader | 训练 shuffle=True、drop_last=True；测试不打乱 |
| 随机种子 | SMOTE：random_state=42；重复实验时可统一设置网络和数据加载种子 |

类别权重在 SMOTE 之后按 `N/(3*n_c)` 计算；当三个类别已完全平衡时，权重均为 1。训练数据由 DataLoader 随机打乱后分批加载。

## 6. 如何运行

### 6.1 运行前必须确认

运行前准备好六个输入 CSV，并确认所需依赖能够正常导入。

程序将结果保存至工作目录。重复运行前请另存已有的同名权重、图和结果 CSV。

### 6.2 运行 Deformable MSCNN

在该模型目录放好六个 CSV 后执行：

```powershell
Set-Location -LiteralPath '文件路径'
& '解释器路径' -X utf8 .\train_predict.py
```

运行其他模型时，将工作目录切换为 `CNN`、`ResNet`、`MSCNN` 或 `SE+MSCNN`，执行同样入口即可。不要在项目根目录直接运行各子目录入口并假定数据路径会随脚本变化。

控制台会显示 SMOTE 后分布、类别权重、训练/测试损失、总体准确率及各类别统计。结束训练后保存训练曲线并调用 `plt.show()`；交互后端下需要关闭图窗，程序才继续保存权重并导出预测。无图形界面时可在运行前设置 `$env:MPLBACKEND='Agg'`。

### 6.3 轮换测试井

当前只执行一组 1244E+1247B→1252A。若要分别让三口井作为留出井，需要手动设置并独立运行：

| 轮次 | TRAIN_WELLS | TEST_WELL |
|---|---|---|
| 1 | 1244E、1247B | 1252A |
| 2 | 1244E、1252A | 1247B |
| 3 | 1247B、1252A | 1244E |

每轮手动设置训练井和测试井并独立运行，记录模型、划分、随机种子、数据版本、参数和输出。结果整理时分别标记训练井预测和测试井预测。

## 7. 输出说明

| 输出 | 内容 |
|---|---|
| `training_results.png` | 上图为训练/测试损失，下图为测试总体准确率 |
| `hydrate_cnn_model.pth` | `state_dict` 参数与缓冲区；所有模型包括 ResNet 都使用此文件名 |
| `1244E_prediction_results.csv` | 1244E 逐深度预测 |
| `1247B_prediction_results.csv` | 1247B 逐深度预测 |
| `1252A_prediction_results.csv` | 1252A 逐深度预测 |

结果列为 `Depth,True_Label,Predicted_Label`，保持输入行顺序。加载 `.pth` 权重时，使用对应的模型定义，并保持特征顺序、归一化方式和标签字典一致。

## 8. 如何应用到新井

### 8.1 有标签的新井参与训练/评估

准备同格式文件；在 `WELLS` 中加入或替换井号，并同步修改 `TRAIN_WELLS`、`TEST_WELL`。所有列、单位、标签含义应与既有数据一致；训练井必须有标签。修改特征数或类别数后需要同步更新配置并重新训练，旧权重通常不兼容。

### 8.2 GPU 应用

当前项目默认基于CPU实现，但可随时调整至GPU训练。使用 GPU 时，在代码中将模型、每批 inputs/labels 和损失函数的 class_weights 放在同一 device；训练、验证和逐井预测均使用该设备。输出转换为 NumPy 前调用 `.cpu().numpy()`。

