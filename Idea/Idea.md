# 模型思路

**视频分支 trustfusion/modules/video_detector.py:26**

- **使用模型**：轻量 3D CNN backbone（Conv3d→AdaptiveAvgPool3d→Linear）+ 两条全连接头（分类与 rPPG）。
- **选择原因**：易于在 Celeb-DF 预处理好的 16 帧 clip 上快速迭代，能同时产出真假 logits 和生理信号，用作后续同步/融合的上下文。
- **局限性**：容量有限，缺乏多尺度与残差结构；rPPG 头只是线性映射，对动作伪迹敏感；若要在真实比赛级别数据上泛化，还需换更强 backbone（如 Xception、TimeSformer）并配合正则与数据增强。

**音频分支 trustfusion/modules/audio_detector.py:26**

- **使用模型**：双向 GRU（默认 2 层、隐藏 256）处理 (B, T, F) 序列，平均池化后接全连接分类头与 spectral_head。
- **选择原因**：RNN 适于捕捉时间依赖；以 log-mel 段为输入能快速构建占位管线，spectral_flow 还能在融合时提供细粒度频谱线索。
- **局限性**：缺乏卷积/自注意力前端，对高频伪造模式把握不足；mel 维度固定（80 bin），对其他谱型需重新适配；无注意力或掩码，遇到缺失帧或噪声段时鲁棒性有限。

**同步分支 trustfusion/modules/sync_transformer.py:37**

- **使用模型**：冻结的预训练 ViT（来自 E:\CUHK\Industrial_Project\vit_model）编码帧级特征，再与线性投影的音频序列相加，经 1 层 TransformerEncoder 聚合，输出 joint_state 与同步 logits。
- **选择原因**：利用强大的 ViT 视觉特征减少训练负担；Transformer 可以建模跨模态的时间依赖；输出联合向量便于后续融合。
- **局限性**：ViT 体积大、推理耗时，对分辨率和归一化敏感；只用一层编码器，表达力有限；假设视频与音频严格对齐，未处理缺帧/时间漂移；默认冻结参数，如需迁移得考虑微调策略。

**融合层 trustfusion/modules/fusion_model.py:34**

- **使用模型**：先拼接 rppg_trace、spectral_flow、joint_state 做线性映射 + 两层 MLP，再与三路 logits 合并，通过 nn.Linear 得到最终 logits，同时返回联合嵌入。
- **选择原因**：比单纯线性加权更有表达力，又不会增加太多复杂度；隐向量保留模态语义，有利于解释和下游扩展。
- **局限性**：仍需所有模态输入齐备；隐向量长度不同步时需额外对齐；缺乏自适应 gating，不能动态降权异常模态；当前 MLP 较浅，对高度非线性关系的建模仍然有限。

整体上，这套设计强调“先打通流程、再逐步增强”。若要进一步提升性能，需要：更强 backbone 与自适应正则、同步模块的多层注意力或 LoRA 微调、融合层加入模态置信度估计/注意力机制，以及对缺失模态的鲁棒处理。