# 智能家居设备分类（测试覆盖面）

综合三处来源整理成 15 类，每类至少一台设备出现在 `homes/` 里：

- Home Assistant 可控实体域：light、switch/plug、climate、cover、fan、lock、media_player、vacuum、
  humidifier、water_heater、valve、alarm_control_panel、camera、doorbell 等
  （[integrations](https://www.home-assistant.io/integrations/)）
- Matter 设备类型：1.0 的照明、插座、温控、窗帘、门锁；1.2 新增冰箱、洗碗机、洗衣机、空气净化器、
  房间空调、扫地机器人、风扇；1.3 新增微波炉、干衣机；1.4 扩展到热泵、储能等能源设备
  （[CSA Matter 1.2](https://csa-iot.org/newsroom/matter-1-2-arrives-with-nine-new-device-types-improvements-across-the-board/)、
  [matter-smarthome 设备类型](https://matter-smarthome.de/en/development/these-device-types-are-available-in-the-matter-standard/)）
- 米家品类：安防、照明、插座开关、传感器、影音娱乐、环境电器、厨房电器、卫浴、清洁电器、宠物与植物等
  （[米家](https://home.mi.com/)）

| 类别 | apartment | house |
|---|---|---|
| 照明 | 客厅主灯、主卧灯、床头灯 | 书房灯、餐厅吊灯 |
| 插座开关 | — | 饮水机插座 |
| 空调温控 | 客厅空调、主卧空调 | 地暖 |
| 风扇 | — | 落地扇 |
| 空气环境 | 空气净化器、加湿器 | 新风机 |
| 窗帘遮阳 | 客厅窗帘、电动晾衣架 | — |
| 影音娱乐 | 电视、智能音箱 | 投影仪 |
| 清洁电器 | 扫地机器人、洗衣机 | 洗碗机 |
| 厨房电器 | 电饭煲 | 冰箱、烤箱、油烟机 |
| 热水卫浴 | 热水器 | 浴霸 |
| 安防门禁 | 智能门锁、摄像头 | 门铃、安防警报、车库门 |
| 能源 | — | 充电桩 |
| 园艺水路 | — | 花园浇水阀 |
| 宠物 | — | 宠物喂食器 |
| 传感器 | 温湿度计（只能查询） | — |

`villa` = 两户合并，36 台设备。
