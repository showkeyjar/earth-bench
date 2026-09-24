# EarthBench 灾种扩展与人类影响建模方案（草案 v1）

> 目标：在不破坏现有评测纪律的前提下，把灾难类型从「窄而深」扩展为「宽而深」，
> 并补上「对人类活动与安全影响」的暴露/脆弱性建模层。
>
> 本文档是设计蓝图，不是已实现功能的描述。文中所有标准编号与阈值数字
> 在落地实现时必须逐一锚定规范原文；经验性阈值一律走「披露假设」通道
> （沿用 `cars_impact_models.json` 中 wind 触发成本比 0.3 的披露先例）。

---

## 0. 背景与差距

现状（截至本草案）：

| 维度 | 现状 | 差距 |
|---|---|---|
| 灾种 | fire / flood / drought / heat 四类 + ecology 占位 | 缺台风、寒潮、暴雪、滑坡等高安全影响灾种 |
| 人类影响 | impact-first 触发（湿球 ≥27℃、35/37℃ 高温线、按城门控低温） | 无暴露/脆弱性维度，同等灾强不区分城区/无人区 |
| 经济评分 | 全局 cost ratio（0.3 为主） | 无分行业损失模型 |
| 决策模板 | 5 模板已定义，仅 ALERT 被基准化 | Dispatch/Upgrade/Close/Recover 无用例 |
| CARS 概率层 | heat / tp / wind 三通道 | 无台风级大风档、无降雪通道 |

## 1. 设计不变量（扩展过程中不许破坏）

1. **真值独立**：新灾种 Ground Truth 一律「外部标准查表」推导，与
   `agents.py` 的线性加权评分完全脱钩，杜绝自证循环（`scenarios.py`
   头注已确立的纪律）。
2. **Impact-first**：触发只认人类伤害阈值；相对异常（+2σ）仅作严重度
   背景。新灾种同理（例：北方冬季常态低温 ≠ 寒潮，必须有「多窗口降温
   幅度 OR + 绝对低温」AND）。
3. **数据可得性优先**：新灾种必须可由现有通道推导（GEFS：t2m / tmax /
   tmin / APCP / UGRD / VGRD10；QWeather；FIRMS），否则进「远期」池。
4. **对抗性配套**：每个新灾种交付基础用例的同时，必须交付误报向 +
   漏报向各 ≥1 个对抗用例，保证规则 baseline < 100%（判别力可测）。
5. **渐进兼容**：`ScenarioCategory` 枚举、`MultiAlertAgent` 路由、
   JSON 场景格式向后兼容；`exposure` 等新字段可选、默认缺失不改变现行为。

## 2. 扩展总览：三层

```
层 1  灾种广度   +滑坡泥石流 +台风大风 +寒潮冰冻 +暴雪（远期：大雾、空气质量）
层 2  暴露脆弱性 ExposureProfile：人口/用地/关键设施/脆弱人群 → 调制动作等级
层 3  决策闭环   Dispatch/Upgrade/Close/Recover 模板套件 + 分行业 cost-loss
```

优先级排序依据 = （对中国人类安全的边际影响 × 与现有数据通道的耦合度）÷ 实现成本。

---

## 3. 层 1：新灾种设计

### 3.1 滑坡/泥石流（landslide）— 优先级 A1（最高）

**为什么第一**：与现有降雨通道（`rainfall_24h` / `rainfall_6h` /
`soil_moisture`）强耦合，是最能复用既有资产的灾种；且是强降雨的
「二次灾害」，天然检验 Agent 的复合推理（雨停 ≠ 风险停）。

**判定变量**：
- 激发雨强：`rainfall_1h`（或 3h 累计）
- 前期有效性：前 3 日有效降雨（指数衰减 API，衰减系数 ~0.8/日）
- 下垫面：`soil_moisture`、静态易发性分区（slope/lithology 图层）
  → `susceptibility_class ∈ {high, mid, low}`

**查表真值（草案数字，锚定规范后固化）**：
参照自然资源部—中国气象局地质灾害气象风险预警四级体系（蓝/黄/橙/红）：

1. 高易发区：1h ≥ 25mm 或 3h ≥ 50mm → 预警（激发雨强通道）
2. 任意易发区：前 3 日有效降雨 ≥ 100mm 且土壤 ≥ 0.80 → 预警（累积饱和通道）
3. 高易发区：24h ≥ 50mm（国标暴雨线）且土壤 ≥ 0.75 → 预警（复合通道）
4. 抑制：连续 48h 无雨且土壤 < 0.50 → 不预警（雨停退坡期，体现「滞后」）

**数据可得性**：APCP 6h 窗可近似短时雨强（披露口径差异）；土壤湿度需在
`cars_serve` 增拉 GEFS 变量，或以前期降雨指数替代（披露）；易发性分区
需静态图层来源（开放地形数据 slope 分级，披露版本号）。

**对抗用例设计**：
- 误报向：雨强达标但易发性 low（线性模型只看雨，会误报）
- 漏报向：当日无激发雨但前期累积 + 饱和 + 高易发（只看「今天」的模型会漏报）

### 3.2 台风/大风（typhoon / windstorm）— 优先级 A2

**现状复用**：wind 冲击通道已有 6/7 级阈值（10.8 / 13.9 m/s 日均口径）。

**扩展**：按 GB/T 19201-2006《热带气旋等级》蒲福氏分档补高档位
（8 级 17.2 = 热带风暴下限；10 级 24.5；12 级 32.7），并增加**阵风口径**
（GEFS pgrb2a 有 GUST 场）：

1. 日均风 ≥ 17.2 m/s（8 级）→ 预警（交通停运/户外作业停止线）
2. 阵风 ≥ 24.5 m/s（10 级）→ 预警（临建设施/塔吊安全线）
3. 大风 + 暴雨 ≥ 50mm 复合 → 更高档（风雨耦合，检验 AND 结构）

**对抗**：日均不高但阵风超标（漏报向）；日均高但为常年大风区正常波动（误报向，
需叠加「是否超气候分位」背景因子——注意仍以绝对阈值为触发，气候背景只调制置信度）。

### 3.3 寒潮/冰冻（cold）— 优先级 B1

**设计要点**：把 `cars_cities.json` 的 `cold_alert_active` 运维门控
升级为正式基准判定。按 GB/T 20484-2017《冷空气等级》（阈值以原文为准）：

1. 24h 降温 ≥ 8℃ 且 日最低 ≤ 4℃ → 寒潮预警
2. 更强档（降温幅度更大 / 极值更低）→ 分级升档
3. 北方冬季常态低温但无「降温幅度」→ 不预警（这正是按城门控的物理含义：
   绝对低温低 ≠ 对人的异常事件）

**数据**：需 t2m 时间序列（已有 f048 缓存 + f000 基准），D 日 vs D-1 日
差值即可，无需新变量。

**对抗**：绝对温低但降幅 3℃（误报向）；降幅 9℃ 但基础温度高、绝对温 8℃（漏报向）。

### 3.4 暴雪/道路结冰（snow）— 优先级 B2

**数据代理**：APCP × (t2m < 0.5℃) 冻结掩膜 → 降雪量（明确披露为代理口径）。
GB/T 28592-2012 附有降雪等级分档（24h 降雪量，数字以原文为准）：

1. 24h 降雪达暴雪档 → 预警
2. 降雪 ≥ 中雪档 且 路温 ≤ 0℃ → 道路结冰复合预警（AND 结构）

**对抗**：降雨 40mm 但全程 t2m > 1℃（线性模型看降水量会误报）；降雪量刚过
小雪线但低温极低 + 湿度大（复合结冰，漏报向）。

### 3.5 远期池（本期不做）

- **大雾**（能见度）：GEFS pgrb2a 0p50 当前子集无 VIS 变量，需扩 GRIB
  变量或 QWeather 实况融合，单独立项。
- **空气质量/霾**：需外部 AQI 数据源与暴露人群模型，超出气象通道范畴。
- **雷电/冰雹**：对流尺度，GEFS 0.5° 分辨率不可支撑，需区域模式。

---

## 4. 层 2：暴露与脆弱性建模

### 4.1 数据结构

`models.py` 新增（全部可选字段，向后兼容）：

```python
class ExposureProfile(BaseModel):
    population_density_class: str = "unknown"   # high / mid / low / none
    land_use: str = "unknown"                   # urban / rural / forest / farmland
    critical_infrastructure: list[str] = []     # ["airport", "hospital", "school"]
    vulnerable_group_ratio: float | None = None # 0-1
    outdoor_activity_level: str = "unknown"     # high(假期/农忙) / normal / low

class ScenarioContext(BaseModel):
    ...
    exposure: ExposureProfile | None = None     # 缺省 → 现行为完全不变
```

### 4.2 关键设计决策：暴露调制动作，不改写物理真值

暴露不改变「这个物理事件是否达到危险等级」（fire-l1 的 FWI 52 在无人区
依然是极高火险），而是改变「该采取哪个层级的动作」：

```
物理等级（查表，独立） × 暴露分级（查表，独立） → 动作矩阵（Alert/Dispatch/Upgrade）
```

动作矩阵本身也查表化（如：高暴露城区 + 暴雨级 → Dispatch；低暴露林区 +
同暴雨级 → Alert），保持真值独立纪律。若国标无对应分级表（城市内涝防治
标准 GB 51222 的重现期分级可作锚点，需核对），则整表作为披露假设发布。

### 4.3 评分联动

`ValueEvaluator` 的期望损失按暴露分级加权：同等命中率，高暴露场景的
漏报罚分更重（V 分数值下降更多），使「在哪儿对了」和「对了多少」一样重要。

---

## 5. 层 2.5：分行业 cost-loss

`eval.py::DEFAULT_COST_RATIO` 从 `{category: ratio}` 扩展为：

```python
{category: {"health": r1, "transport": r2, "agriculture": r3, "energy": r4}}
```

首批全部走披露假设（0.1–0.5 量级），并在报告中逐条列出；后续用历史
灾损数据（应急管理部门公开年报）校准。`ValueEvaluator` 保持旧接口兼容，
新 `SectorValueEvaluator` 输出分行业 V。

---

## 6. 层 3：决策模板闭环套件

现状：`templates.py` 定义了 5 模板的最低观测数与解释格式，但
`get_alert_benchmark_suite()` 全部是 ALERT 判定。补齐：

| 模板 | 决策问题 | 查表真值要点 | 新原语 |
|---|---|---|---|
| DISPATCH | 是否预置资源 | 阈值 + 附加条件复合（FWI 高 + 景区 + 假期 → 消防预置；水位逼近 + 降雨持续 → 抢险待命） | AND 复合 |
| UPGRADE | 是否升档 | 黄档已触发 + 更高档阈值命中 / 趋势外推 | 等级比较 |
| CLOSE | 是否封闭（道路/景区） | 危险等级 × 暴露（台风 10 级 + 机场 → 封闭） | 暴露矩阵 |
| RECOVER | 是否解除 | **条件解除后须持续 N 小时**（火险：24h 雨 ≥30mm 且 FWI 降档后 6h 无反弹；洪涝：水位退至警戒下 0.5m 持续 6h） | 持续时间窗 |

RECOVER/CLOSE 引入「持续时间」维度是现有真值函数没有的能力
（目前只有趋势单调性判断），需要新增 `_persist_for(obs, var, cond, hours)`
辅助原语。这也让「雨停就解除」的过早恢复错误变得可测——正是应急管理
里最常见的人类代价来源。

---

## 7. 实施路线图（文件级触点）

### Phase A（灾种扩展，可独立交付）

> **状态**：层 1 灾种广度扩展全部完成 —— A1（landslide）、A2（typhoon）、
> B1（cold 寒潮）、B2（snow 暴雪/道路结冰）均已落地。
> 基准现为 8 灾种 × 40 基础用例 + 16 对抗用例（8 误报 + 8 漏报），
> 全部测试通过（191 passed），规则 baseline 基础满分 / 对抗全败。
> 下一步：无 —— 扩展方案四层（灾种广度 / 暴露脆弱性 / 模板闭环 / CARS 同步）
> 已全部落地。后续迭代见「风险与开放问题」节（行业成本比校准、暴露分级
> 统计口径锚定等）。
> Phase B / C / D 状态详见下方各状态块。
> **追加交付（2025 收尾）**：历史灾例回填验证 —— 七场真实灾害
> （郑州 7·20 / 海河 23·7 / 台风杜苏芮 / 通辽寒潮 / 通辽特大暴雪 /
> 川渝高温峰值日 / 重庆主城山火日）的公开报道观测值灌入独立标准
> 真值函数，8/8 判定与实际现实一致
> （`get_historical_validation_suite` + `scripts/validate_historical.py`
> → `docs/historical-validation.md`，全部数值附来源 URL）。7/8 灾种
> 真值函数已锚定现实。回填产出三项口径发现（见
> `HISTORICAL_VALIDATION_FINDINGS`）：寒潮 24h 口径排查 → 已修齐
> 2017 版标准（开放问题 2c 清偿）；滑坡激发雨强公开不可溯源（开放
> 问题 6）；干旱 SPI/Palmer 指数公开断链——「偏少 90%」相对口径与
> MCI 特旱定性均无法严格标准化为 SPI（秩上界路径 SPI ≤ Φ⁻¹(1/n)
> 已文档化，待可溯源站点纪录出现）。受阻两类（滑坡/干旱）同属
> 「指数/强度类输入无公开存档」失败类。
> **基准卫生批次**：平凡基线（always/never/random，`trivial_agents.py`）
> + 日报「📐 基准体检」段（对抗/模板/暴露判别力矩阵 + 历史锚定状态，
> 离线渲染）+ CI 接入历史验证脚本（退出码语义）。对抗套件上规则
> baseline 0% 低于随机 56%、always 单向 8FP/0FN——互补失败模式证明
> 套件判别力非类别不平衡伪影。
> **寒潮 2017 标准修齐**（开放问题 2c 清偿）：历史回填的口径排查发现
> 真值误用 2006 版三档（寒潮/强寒潮/特强寒潮）——维基「寒潮·中华
> 人民共和国标准」小节核验确认 2017 版为四级体系（寒潮顶档）→
> `infer_cold_ground_truth` 重写为多窗口降幅 OR（24h≥8/48h≥10/
> 72h≥12，序列推导至 4 日）× 日最低 ≤4°C AND；全量回归零决策翻转
> （仅分数归一顶档 0.95），新增 48h/72h 通道单元测试锁定缓慢渗透型
> 寒潮的可捕获性。
> **LLM Agent 评测 harness**（架构中「LLM Agent via CARM」环节落地）：
> `llm_agent.py`（ScenarioPromptBuilder 结构化提示词——观测表 + 判定
> 纪律（查表思维/AND 缺一/常态≠异常）+ 严格 JSON 输出；阈值双模式
> informed/blind，informed 摘要从 scenarios.py 常量单一来源生成，
> 与规则 baseline 同信息类、不泄露判定；parse_llm_decision 三级鲁棒
> 解析 + 解析失败显式计数；CarmBackend 本地 Mustard CARM /
> Qwen3.6-35B OpenAI 兼容无云调用，失败优雅降级；MockLLMBackend
> 确定性离线后端）+ `scripts/evaluate_llm_agent.py`（与日报体检同表
> 口径的 LLM vs 规则 vs 平凡基线对比，报告含首例提示词与原始输出
> 审计段，`docs/llm-eval-<backend>.md`）+ `tests/test_llm_agent.py`
> （12 项全离线）。**首个真实快照**（Qwen3.6-35B，informed，温度 0，
> 全 83 例，0 解析失败）：对抗 **62% vs 规则 0%**（0FP/6FN，保守偏
> 漏）——「推理型 agent 可与线性加权 measurably 分离」的套件设计
> 目标首次被真实 LLM 实证；基础 78%（0/9 同向偏漏）；模板套件
> ~50%（v1 已知局限：alert 协议未解释 dispatch/upgrade/close/
> recover 语义，列为后续工作）。两个集成发现已披露：CARM 的
> /v1/chat/completions 是工具路由端点（判定提示词被误路由到
> search，须直连其底层 llama.cpp 引擎 127.0.0.1:8082）；Qwen3.6
> 思考链会耗尽 token 预算（reasoning_content 截断致 content 空，
> 需 chat_template_kwargs.enable_thinking=false）。
> **门面站整改**（earth-ai.fun / `index.html`）：去花 + 补新功能两线。
> 去花：全站装饰性 emoji 清除（板块标题/导航/按钮/卡片图标，保留
> ✅❌⚠️ 状态语义与 favicon）、分享按钮与发布徽章扁平化；信息密度
> 提升——四张渐变灾种卡换成 8 行阈值表（数字与 scenarios.py 常量
> 一致：阵风 24.5、8 级风 17.2 等），泛泛的「难度体系」板块删除。
> 补新功能：新增「基准矩阵」板块（83 用例分层构成表 + 对抗 16 例
> 判别力表（规则 0% / LLM 62% / 平凡基线）+ 7 场历史锚定事件表 +
> LLM 赛道 informed/blind 双口径说明，链接 historical-validation.md
> 与 llm-eval-carm.md）；SEO/JSON-LD 同步（8 灾种 83 用例，删
> 「全球首个」浮夸语与「四类灾种 20 用例」过时计数）；修正过时
> 声明（Qwen3 14B via Ollama → Qwen3.6-35B 本地 llama.cpp）。
> `scripts/check_index_html.py` 结构校验（标签配对/锚点/内容令牌/
> 过时令牌）全绿。
> **旱灾影响门控**（用户指出：缓发灾种「气象干旱成立 ≠ 影响成立」，
> 现代城市短时/局地干旱无决策价值）：`infer_drought_ground_truth`
> 重写为两通道 AND——指数通道（GB/T 20481 SPI/Palmer 定档，逻���
> 不变）× 影响门控（持续 ≥60 天 / 受旱面积 ≥40% / 城市缺水率
> ≥10%，SL 424-2008 城市中度线披露口径；缺影响观测视为未验证
> → 不预警）。指数红但影响不成立 → False 0.30。基础 3 个 True
> 用例补影响观测（CMA-DroughtBulletin / WaterAuthority 来源）；
> 对抗两例重设计：`adv-drought-urban-short-no-impact`（误报陷阱：
> SPI -2.6 全指数红、持续 22 天/面积 12%/缺水 4% 全未达标 → GT
> False，规则只读指数 0.827 → 误报）与 `adv-drought-impact-gated`
> （漏报陷阱：SPI -1.2 中旱 + 95 天 + 55% 面积 → GT True 0.75，
> 影响观测对规则 agent 不可见，0.130 → 漏报）。回归后对抗仍
> 16 例规则 0%（8FP/8FN）、基础 100%、GT 零分歧；新增 3 项门控
> 单元测试（城市短旱/缺观测/中旱长旱）→ 250 通过。README 新增
> 门控段落、干旱表行更新；门面站八类表干旱行、LLM informed
> 阈值摘要、干旱历史回填 finding 同步补门控口径。
>
> B2 交付物：`models.py`（SNOW 枚举）、`scenarios.py`（GB/T 28592-2012
> 降雪三档查表 + 道路结冰「中雪档 × 路温」AND 复合 + 冻结掩膜代理
> （snowfall 缺失时由 rainfall × t2m<0.5℃ 推导，雨不计数）、5 基础用例、
> 2 对抗用例）、`agents.py`（SnowAlertAgent 线性 baseline + 路由）、
> `eval.py`（snow 成本比 0.3 披露）、`tests/test_snow.py`。
> 规则 baseline：基础 5/5 满分、对抗 0/2（雨非雪误报——冻结掩膜缺失；
> 结冰双线刚过漏报——AND 稀释）。
>
> B1 交付物：`models.py`（COLD 枚举）、`scenarios.py`（GB/T 20484-2017
> 查表真值 + 降幅序列推导、5 基础用例、
> 2 对抗用例）、`agents.py`（ColdWaveAlertAgent 线性 baseline +
> MultiAlertAgent 路由）、`eval.py`（cold 成本比 0.3 披露）、
> `tests/test_cold.py`。规则 baseline：基础 5/5 满分、对抗 0/2
>（常态低温无降幅误报 + 双线刚过被稀释漏报）——判别力成立。
> **后修（历史回填发现）**：初版误用 2006 版「寒潮/强寒潮/特强寒潮」
> 三档结构，已修齐为 2017 版四级体系顶档：多窗口降幅 OR（24h≥8 /
> 48h≥10 / 72h≥12）× 日最低 ≤4°C AND；全部用例零决策翻转，仅分数
> 归一顶档 0.95（`tests/test_cold.py::test_cold_multiday_window_or`
> 锁定 48h/72h 通道）。
> 基准层已把 cars_cities.json 的 cold_alert_active 按城门控升华为正式
> 判定（绝对温低 ≠ 对人的异常，必须叠加降幅）；CARS 服务层的低温通道
> 口径区分已在 Phase D 收尾文档化（冻害 ≠ 寒潮，跨日降幅集合配对列为
> 开放问题，见「风险与开放问题」2b）。
>
> A2 交付物：`models.py`（TYPHOON 枚举）、`scenarios.py`（四级风档+阵风
> 安全线+风雨耦合查表真值、5 基础用例、2 对抗用例）、`agents.py`
>（TyphoonAlertAgent 线性 baseline + MultiAlertAgent 路由）、`eval.py`
>（typhoon 触发成本比 0.3 披露假设）、`data_collectors.py`（温州/海口/
> 厦门区域坐标）、`tests/test_typhoon.py`。规则 baseline：基础 5/5 满分、
> 对抗 0/2（次阈值混合误报 + 阵风稀释漏报）——判别力成立。
> CARS 概率层的 wind 高档位（17.2/24.5）与 GUST 通道属 Phase D1，待实施。
>
> A1 交付物：`models.py`（LANDSLIDE 枚举）、`scenarios.py`（三通道+退坡
> 抑制查表真值、5 基础用例、2 对抗用例）、`agents.py`（LandslideAlertAgent
> 线性 baseline + MultiAlertAgent 路由）、`eval.py`（landslide 触发成本比
> 0.3 披露假设）、`data_collectors.py`（5 个滑坡区域坐标）、
> `tests/test_landslide.py`（真值一致性 + 通道单元行为 + baseline 双向失败）。
> 规则 baseline：基础 5/5 满分、对抗 0/2（误报+漏报各一）——判别力成立。

| 步骤 | 文件 | 内容 | 状态 |
|---|---|---|---|
| A1 | `earthbench/models.py` | `ScenarioCategory` 加 `LANDSLIDE / TYPHOON / COLD / SNOW`，同步 `_CATEGORY_MAP` 与 `from_string` 告警文案 | 全部 ✅ |
| A2 | `earthbench/scenarios.py` | 4 个 `infer_*_ground_truth()` 查表函数；每灾种 5 个基础用例（L1–L4）+ 2 个对抗用例（误报/漏报各一） |
| A3 | `earthbench/agents.py` | 4 个 Rule Agent（线性加权 baseline，供对抗用例「打败」）；`MultiAlertAgent.category_map` 注册，未知类别不再静默回退 fire |
| A4 | `tests/` | 仿 `test_adversarial.py`：断言真值函数输出 == 用例标签、规则 baseline 在对抗用例 < 100% |
| A5 | `README.md` | 场景类别表、对抗套件说明更新 |

### Phase B（暴露 + 行业 V）

> **状态**：层 2 暴露脆弱性建模已实施落地（203 tests passed）。
>
> 交付物：
> - `models.py`：`ExposureProfile`（人口/用地/关键设施/脆弱人群/户外活动，
>   全可选）+ `ScenarioContext.exposure`（缺省 None，40 个既有用例行为不变）
> - `scenarios.py`：`infer_exposure_class`（E0-E3 查表，披露假设）、
>   `infer_action_ground_truth`（物理等级 × 暴露分级 → 动作矩阵：
>   monitor/alert/dispatch）、`get_exposure_suite`（10 用例：4 对同观测
>   异暴露配对 + 1 不变量 + 2 动作级陷阱）
> - `agents.py`：`MultiAlertAgent.decide_action`（线性置信度 ≥ 0.85 作
>   严重度代理，刻意与真值 0.90 分界不同 → 动作级判别力）
> - `benchmark.py`：`evaluate_action_agent`（动作级评测）、exposure 透传、
>   行内 exposure_class
> - `eval.py`：暴露加权 V（可选开关，E3 漏报 1.5×，披露）+
>   `SectorValueEvaluator`（health/transport/agriculture/energy 分行业
>   成本比，全部披露假设，未知行业回退默认）
> - `tests/test_exposure.py`（12 项：向后兼容 / 分级查表 / 矩阵不变量 /
>   配对性 / baseline 基础 40/40 满分 + 暴露套件 8/10 双向出错 / V 加权）
> - CLI：`python -m earthbench --benchmark --exposure`
>
> 关键不变量验证：物理 NO × 任意暴露（含 E3）→ monitor（暴露不创造风险）；
> 同观测异暴露配对仅动作真值不同、物理真值相同。
> baseline 动作准确率：基础套件（E0）40/40，暴露套件 8/10——欠响应
>（severe×E2 线性置信度 0.67）与过响应（moderate×E2 线性置信度 0.86）
> 各一，动作维度的判别力成立。

| 步骤 | 文件 | 内容 | 状态 |
|---|---|---|---|
| B1 | `earthbench/models.py` | `ExposureProfile` + `ScenarioContext.exposure` | ✅ |
| B2 | `earthbench/scenarios.py` | 暴露变体用例：同一物理观测 × 两种暴露 → 不同动作真值 | ✅（get_exposure_suite，10 用例） |
| B3 | `earthbench/eval.py` | `SectorValueEvaluator`（兼容旧接口） | ✅（+ 暴露加权 V） |

### Phase C（模板闭环）

> **状态**：层 3 决策模板闭环已实施落地（214 tests passed）。
>
> 交付物：
> - `scenarios.py`：`_persist_for` 持续时间原语（[now-Nh, now] 闭区间、每个
>   采样点满足条件且 >= 2 点才算「持续」）；`_detect_template_category` 从观测
>   变量推断灾种；四个模板真值函数 `infer_dispatch/upgrade/close/recover_ground_truth`
>   （全部披露口径：阈值 + AND 复合 / 等级比较 / 危险×暴露矩阵 / 持续窗）；
>   四个模板套件 `get_dispatch/upgrade/close/recover_suite`（4+4+5+4=17 用例）
> - `agents.py`：`MultiAlertAgent.decide` 按 `context.template` 路由；四个模板
>   baseline 统一用「线性 ALERT 置信度阈值」做启发式，刻意不复制真值的 AND 与
>   持续窗结构 → 每个套件误报 + 漏报陷阱各一
> - `benchmark.py`：`AlertTestCase.template` 字段 + 通用签名 inspect 传参
>   （heat_duration_days / exposure 按函数签名注入）
> - `tests/test_templates.py`（11 项：原语单元 / 真值一致性 / 判别力双向 / 路由）
> - CLI：`python -m earthbench --benchmark --template {dispatch|upgrade|close|recover}`
>
> 判别力验证（baseline 每套件 50%-60%、误报漏报双向）：
> - DISPATCH：FWI 45 无人林区误报（暴露 AND 缺失）+ 水位降雨双达线漏报（AND 稀释）
> - UPGRADE：8 级 + 阵风 20 未达 10 级线误报（等级比较缺失）+ 高温三连升漏报（趋势外推缺失）
> - CLOSE：10 级风开阔海面误报（机场矩阵缺失）+ 暴雪路温刚过冰点漏报（暴露阈值）
> - RECOVER：**过早恢复**（水位 6h 前仍 4.7，雨停即解除）+ **过度保守**
>   （路温已持续解冻仍不解除）——前者正是应急管理最常见人命代价来源。

| 步骤 | 文件 | 内容 | 状态 |
|---|---|---|---|
| C1 | `earthbench/scenarios.py` | `get_dispatch_suite()` / `get_close_suite()` / `get_recover_suite()` + 持续时间原语 | ✅（含 upgrade，共 4 套件 17 用例） |
| C2 | `earthbench/benchmark.py` | 按 template 过滤 | ✅（`AlertTestCase.template` + 签名 inspect 传参） |
| C3 | `earthbench/__main__.py` | CLI `--template` 参数 | ✅ |

### Phase D（CARS 概率层同步）

> **状态**：已实施落地（217 tests passed）——扩展方案四层全部完成。
>
> 交付物：
> - D1 `cars_serve_impact.py`：wind 高档位 `p_harm_wind_typhoon`（≥17.2 八级
>   交通停运线）/`p_harm_wind_extreme`（≥24.5 十级临设参考档，日均口径）；
>   snowfall 派生通道 `snowfall_members`（APCP × 冻结掩膜 t2m<0.5°C，与基准
>   暴雪真值同口径）+ `fetch_ops_t2m_mean`（同批 grib 加读 2m 气温，零额外
>   下载）→ `p_harm_snow`/`p_harm_snow_intense`（10/20mm 水当量）
> - D2 `cars_impact_models.json`：wind thresholds 加 typhoon/extreme；新增
>   snow 配置节（derived_from tp、freeze_mask_c、披露 provenance）
> - D3 `cars_verify_impact.py`：新阈值档检验口径——`obs_gefs_t2m_valid_day`
>   （冻结掩膜观测）+ wind_typhoon/snow 的 Brier/命中漏报/CSI/可靠性分箱
>   （`_var_summary` 泛化为四变量，旧历史缺字段优雅降级）
> - 面板：暴雪 ❄️/台风级 🌀 旗标与检验摘要行（publish_pipeline）
> - `SNOW_P_TRIG` 与 eval snow 成本比构造性同步（单一来源）
>
> 口径一致性（tests 断言兜底，杜绝文档-代码漂移）：
> - wind typhoon/extreme == 基准 WIND_LEVEL_8/WIND_LEVEL_10（17.2/24.5）
> - snow hard/intense == 基准 SNOW_BLIZZARD/SNOW_HEAVY_BLIZZARD（10/20）
> - freeze_mask_c == 基准 SNOW_FREEZE_MASK_C（0.5）
> 披露（无法由本通道服务的口径）：GUST 阵风安全线（无阵风数据，日均弱代理）；
> 冻结掩膜为未扰动 c00 确定性判定（雨雪相位不确定性未入集合）。

| 步骤 | 文件 | 内容 | 状态 |
|---|---|---|---|
| D1 | `earthbench/cars_serve_impact.py` | wind 通道加 17.2 / 24.5 档；新增 snowfall 通道（APCP × 冻结掩膜） | ✅ |
| D2 | `earthbench/data/cars_impact_models.json` | 阈值与披露假设同步 | ✅ |
| D3 | `earthbench/cars_verify_impact.py` | 新阈值档的验证口径 | ✅（四变量泛化 + 冻结掩膜观测） |

### 验收标准（每个 Phase 通用）

1. `python -m pytest tests/ -v` 全绿；
2. 新灾种 ≥5 基础 + ≥2 对抗用例，对抗用例上规则 baseline 判别力 < 100%；
3. 真值函数只引用外部标准/披露假设，不 import `agents.py` 的权重；
4. README / docs 表格同步，无「文档宣称但代码没有」的落差。

---

## 8. 风险与开放问题

1. **标准锚定**：草案中的阈值数字（尤其滑坡经验阈值、降雪分档）是
   占位，实现时必须逐条对照规范原文或文献，查不到的一律进披露假设
   清单。（寒潮强档占位已清偿：见 B1「后修」——2006 三档结构误用经
   历史回填发现后修齐为 2017 四级体系。）
2. **GEFS 口径近似**：6h 窗 APCP 近似 1h 雨强、冻结掩膜近似降雪，口径
   差异要在披露文档与验证报告中显式量化。
   2b. **寒潮通道暂不服务**：CARS p_harm_cold 是冻害口径（日最低 ≤ 0°C），
   基准寒潮真值是「多窗口降幅 OR × 日最低」AND——跨日降幅需 2-4 日
   集合配对，已在 cars_serve.py / cars_agent.py 披露区分，列为后续工作。
   2c. **寒潮真值 vs 地方预警口径**（历史回填发现，已部分清偿）：
   排查发现真值误用 2006 版三档结构 → 已修齐 2017 版四级体系顶档 +
   多窗口 OR（24h≥8/48h≥10/72h≥12 且 ≤4°C，全量回归零决策翻转）。
   修齐后 2016年1月寒潮广州/上海站档仍不达国标线（48h 降幅 5.8/9.3
   均低于 48h 窗口 10°C 线），但地方实发了红色寒冷预警（绝对低温口径）/
   蓝色寒潮预警（48h 口径）——残余分歧是国标（过程强度分级）vs 地方
   标准（民生影响）的口径差异，属披露而非缺陷，见
   `HISTORICAL_VALIDATION_FINDINGS`。
6. **指数/强度类输入无公开存档的灾种回填受阻**（历史回填发现，两类）：
   滑坡——舟曲 2010 激发雨强定量值公开渠道不可溯源（中英文条目均无
   mm 值，县站存档缺失），需从文献取数（Ren et al. 2014 JGR:
   Atmospheres, doi:10.1002/2013JD020881）；干旱——SPI/Palmer 指数
   站档不含、公开报道只有「偏少 90%」相对口径与 MCI 特旱定性，
   无法严格标准化（需站点气候基线 σ）。若出现可溯源站点纪录
   （如「8 月降水为 1951 年以来最少」），可用秩上界推导：
   SPI ≤ Φ⁻¹(1/n)（n=61 → ≤ -2.42 特旱档）。按「宁缺毋滥」纪律
   均不构造无法溯源的用例。
3. **静态图层依赖**：滑坡易发性分区（slope/lithology）需要确定开放数据源
   与版本冻结策略（对齐 CARS「冻结模型 + 披露」的先例）。
4. **暴露分级表的规范锚点**是最弱一环：很可能没有现成国标，届时整表按
   披露假设发布，并在对抗用例中测试其对分级错误的敏感度。
5. **灾种膨胀 vs 用例质量**：每加一个灾种约 7 个用例的维护成本，坚持
   「宁缺毋滥」——没有对抗判别力的灾种不进基准。
