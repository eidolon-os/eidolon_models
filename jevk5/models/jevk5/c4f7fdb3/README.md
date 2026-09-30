# 固定 JevK5-4B 资产

模型来源：[alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5)，固定 revision `c4f7fdb3aeab5582336406e78d3bef11bf98833d`。上游模型卡声明 Apache-2.0，基于 Qwen3.5-4B，已合并其原始决策适配权重。本项目后续训练的是这份候选权重上的新适配器，不是从纯预训练基座开始。

运行代码来源：[allebee/jevk5](https://github.com/allebee/jevk5)，固定 commit `1e5ae1b533b9eb80c0cbe3fbd010607d0b4e26ae`，代码许可证与来源清单在项目 `src/eidolon_models_jevk5/vendor/jevk5/`。

`manifest.json` 记录全部8个本地资产文件的SHA256。权重文件摘要为 `13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c`。`weights/` 不入Git；当前通过硬链接复用历史已校验缓存，视为不可变，不能覆盖写入或在该目录保存训练结果。

本轮为 PyTorch fp16 / Mac MPS 本地决策参考实现，无量化转换。最多16选项，输出选项概率，不生成回复文本。新适配器保存在独立 runs 目录并绑定基座摘要。中文角色调度语义与目标设备速度均须以本项目实验结果为准，不能套用上游其他硬件的速度数据。
