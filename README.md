# ZzkMind

一个基于 PyTorch 的轻量级大语言模型学习项目，参考 MiniMind，从零开始实现最小可运行的训练与推理流程。

## 当前目标

- 完成项目基础环境搭建
- 实现 YAML + dataclass 配置管理
- 实现最小 tokenizer、dataset、model、trainer
- 支持训练和文本生成

## 项目结构

```text
ZzkMind/
├─ configs/
├─ dataset/
├─ model/
├─ trainer/
├─ scripts/
├─ out/
├─ config.py
├─ main.py
├─ pyproject.toml
└─ README.md