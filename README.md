# gme-qwen2-vl-for-generate


目的：基于gme-qwen2-vl原多模态向量模型，实现文本生成功能

1. 下载gme-qwen2-vl模型
2. 替换模型文件中的gme_inference.py，主要改动点：
- 增加generate方法，用于生成文本
- 增加embedding多种池化策略，用于衡量多种向量提取的结果表现