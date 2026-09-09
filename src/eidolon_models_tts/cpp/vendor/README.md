# NPU 运行时头文件（vendored）

这两个头文件是链接板子 NPU 运行时所需的 ABI 声明。`.so` 是主机自己的（属于
板卡的 vendor 包），头文件在编译期需要——而它们原来只是手工放在板上的
`/root/npu` 和 `/root/rkllm` 里，`eidolon` 用户根本读不到。发布因此在真机上以
`fatal error: rkllm.h: No such file or directory` 失败了一次。

所以它们入库，按不可变的上游 commit 钉住，并逐字节和板上运行的那一份比对过。

| 文件 | 上游 | 来源 | sha256 |
|---|---|---|---|
| `rknn_api.h` | airockchip/rknn-toolkit2 | `42aa1d426c0a9e0869b6374edba009f7208a1926` 的 `rknpu2/runtime/Linux/librknn_api/include/rknn_api.h` | `c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a` |
| `rkllm.h` | airockchip/rknn-llm | `878f9361fd3afa7e167b7079918918f78d2c1c2a` 的 `rkllm-runtime/Linux/librkllm_api/include/rkllm.h` | `80596a578f7f8e70df6eda1c2cbead3bfced14623a190258f2bd009a3d1f72cf` |

两个 commit 都是把分支/标签解析出来的结果，不是分支本身——分支可以在 digest 底下
被移动，那时失败的是 pin，而不是文件被发现不对。`rknn_api.h` 的 commit 和
`ops/component.toml` 里为 `librknnrt.so` 2.3.2 记的是同一个。`rkllm.h` 取自
`release-v1.3.0`（板上 `librkllmrt.so` 报的运行时版本正是 1.3.0），注意仓库里
另有一个 `v1.3.0` 标签指向别的东西。

**头文件必须和主机上的 `.so` 版本对得上。** 升级板卡的 NPU 运行时时，这两个也要
一起换，并重新和主机上的副本比对——ABI 不匹配的表现不是编译失败，是运行时行为
不对。
