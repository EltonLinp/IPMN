# Structure_Graph

```Mermaid
graph TD
    A[Input data] --> B1[Video branch: 
    3D CNN→AdaptiveAvgPool3d→Linear
    Output: True and false logits, rPPG physiological signals
    Features: Lightweight and fast; Sensitive to motion artifacts]
    A --> B2[Audio branch: 
    Bidirectional GRU （2 layers, 256 hidden units） → Average pooling → Fully connected classification head + spectral_head
    Output: True and false logits, spectral_flow spectral features
    Characteristics: Skilled in time series modeling; Limited high-frequency identification]
    A --> B3[Synchronous branch: 
    Freeze ViT （visual features） + linear projected audio sequence → Aggregate by TransformerEncoder
    Output: joint_state, synchronous logits
    Features: Strong cross-modal dependence; It takes a relatively long time.]

    B1 --> C[Fusion layer: 
    Splicing （rPPG, spectral_flow, joint_state） → Linear → Two-layer MLP → Merging multimodal logits
    Output: Final true and false logits, joint embedding vector
    Features: Strong expressiveness, and the attention mechanism can be further incorporated]
    B2 --> C
    B3 --> C

    C --> D1[Output: Authenticity classification]
    C --> D2[Output: Fusion embedding]

```

