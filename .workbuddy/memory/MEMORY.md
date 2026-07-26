# EarthBench 项目长期记忆

## 推广策略 (2026-07-24)

### 核心定位
EarthBench = 全球首个测试 AI 在地球科学领域"做决策"的基准框架，非"预测精度"。Slogan: Decision > Prediction.

### 宣传矩阵
- **Tier 1**: earth-ai.fun 网站 + 微信公众号"爱智模/语森者" + GitHub
- **Tier 2**: Hugging Face Space、Papers With Code、知乎、CSDN、InfoQ、Bilibili
- **Tier 3**: 学术会议 (NeurIPS D&B, AGU, IGARSS)、行业峰会、政府对接

### 内容节奏
- 每日: 08:00 / 20:00 自动更新日报
- 周一: 趋势预判周报
- 周三: AI 决策案例分析
- 周五: 技术博客

### SEO 配置
- index.html 已添加 meta tags、OG tags、JSON-LD structured data
- sitemap.xml + robots.txt 已创建
- RSS feed + JSON API 已就绪

### 自动化 pipeline
- `.github/workflows/daily-report.yml` — 每日自动运行 publish_pipeline 并部署 GH Pages
- `.github/workflows/auto-post.yml` — 自动生成周报和 GitHub Release
- `scripts/wechat_summary.py` — 从日报生成微信推文摘要
