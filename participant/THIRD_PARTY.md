# 随包资源

仿真 Python 代码、配置和程序化城市场景复制自本工作区 `neon-arena`，选手包使用独立快照。未复制平台账号、权限、数据库、容器服务或预训练策略。

- G1 模型来自 Google DeepMind MuJoCo Menagerie 的 `unitree_g1`，版本 `370c43965323c0944b433b202aac5058b5f907a9`；BSD-3-Clause 许可证及上游说明位于 `third_party/g1/LICENSE`、`third_party/g1/UPSTREAM_README.md`。原始模型不改动，运行时添加双机器人、相机和比赛场景。
- Orbitron 字体来自 google/fonts，版本 `8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5`；SIL OFL 许可证位于 `third_party/orbitron/OFL.txt`。用于程序化城市招牌。
- 逐文件下载地址、版本和 SHA-256 见 `assets.lock.json`，仅列出本包携带的上游资源。
- MuJoCo、NumPy 和 Pillow 由 pip 安装，各自遵循其发行包许可证。

本包不含 Holosoma 或 Unitree RL Gym 的模型权重；基础示例不依赖 ONNX Runtime 或 PyTorch。
