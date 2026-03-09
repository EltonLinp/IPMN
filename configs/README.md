# Audio Configuration Notes

- `configs/audio_preprocess_config.json` 汇总了 FakeAVCeleb 预处理所需参数（数据源、输出目录、采样策略等）。运行时直接加载该文件：
  ```powershell
  python preprocessing/fakeav_preprocessor.py --config configs/audio_preprocess_config.json
  ```
  当前配置会遍历 `FakeAVCeleb_v1.2/` 中所有说话人，只要存在真实或伪造样本就会保留，并分别抽取可用的全部条目（受 `real_per_speaker` 与 `fake_per_speaker` 上限限制；默认给到 9999，等价于“全量”）。结果写入 `O:/processed/fakeav_audio_full/`，若某些样本解码失败会在 `preprocess_index.jsonl` 中留下 `status`，方便后续排查。
- `configs/audio_train_config.json` 管理训练阶段超参，包含 `wav2vec_path`、`wav2vec_trainable` 与 `verify_dataset` 等字段。`wav2vec_path` 默认为 Hugging Face 的 `facebook/wav2vec2-base-960h`，若想改用本地或其他预训练模型，只需更新该字段；保持 `wav2vec_trainable: false` 可冻结编码器。
- 预处理完成后，建议再运行一次轻量校验，确认所有 `.pt` 文件持有 `mel` 与 `waveform`：
  ```powershell
  python training/train_audio.py --config configs/audio_train_config.json --verify-dataset --epochs 0 --num-workers 0
  ```
  该命令沿用训练配置中的 `data_dir` 与 `index_file`，在不进入完整训练循环的情况下完成一致性检查与一次前向评估。
- 正式训练命令：
  ```powershell
  python training/train_audio.py --config configs/audio_train_config.json
  ```
  若需微调 wav2vec，可以在 JSON 中将 `wav2vec_trainable` 改为 `true` 或直接在命令行追加 `--wav2vec-trainable`。训练阶段显存占用较高，必要时可适当调小 `batch_size` 或减少 DataLoader worker。
