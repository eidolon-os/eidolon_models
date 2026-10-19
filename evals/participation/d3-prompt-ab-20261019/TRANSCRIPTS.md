# A/B真实回复文本闭环

两策略独立生成回复；用户后续输入为固定脚本。非生产Agent或设备验收。

## pronoun_clarification · baseline

动作/人选/终态：True

用户：麻烦让那位继续介绍吧。

- clarify:M0：请问您希望哪位继续介绍呢？

终态：clarify

用户：我指的是知夏。请给一个参观摄影展的小建议，一句说完后结束。

- respond:M2：建议你参观摄影展时先整体走一遍，再回头细看最打动你的几张作品。
- finish

终态：finish

## pronoun_clarification · enhanced

动作/人选/终态：True

用户：麻烦让那位继续介绍吧。

- clarify:M0：请问您希望哪位继续介绍呢？

终态：clarify

用户：我指的是知夏。请给一个参观摄影展的小建议，一句说完后结束。

- respond:M2：看展时可以先整体走一圈，再挑最喜欢的几幅细看，留意一下展签上的拍摄背景。
- finish

终态：finish

## missing_options_clarification · enhanced

动作/人选/终态：True

用户：这两个我选哪一个比较好？

- clarify:M0：这两个选项分别是什么，能先告诉我吗？

终态：clarify

用户：我在选画画或跑步，想在室内放松。只请秋岑给一个选择和理由，一句说完结束。

- respond:M0：选画画吧，室内安静又放松，还能慢慢享受动手的乐趣。
- finish

终态：finish

## missing_options_clarification · baseline

动作/人选/终态：True

用户：这两个我选哪一个比较好？

- clarify:M0：这两个选项分别是什么，能先告诉我吗？

终态：clarify

用户：我在选画画或跑步，想在室内放松。只请秋岑给一个选择和理由，一句说完结束。

- respond:M0：选画画吧，室内安静又容易放松，慢慢画也不赶时间。
- finish

终态：finish

## normal_response · baseline

动作/人选/终态：True

用户：只请远帆给一个开始跑步的通用建议，不用询问我的情况，一句话说完结束。

- respond:M1：先换上跑鞋出门慢跑十分钟，比想太多更重要。
- finish

终态：finish

## normal_response · enhanced

动作/人选/终态：True

用户：只请远帆给一个开始跑步的通用建议，不用询问我的情况，一句话说完结束。

- respond:M1：先换上跑鞋，出门慢跑十分钟就行。
- finish

终态：finish
