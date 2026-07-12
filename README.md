# auto-paper-collecter-cloud

🌤️ 云端文献雷达 — GitHub Actions 每日自动抓取、AI 筛选、中文摘要、微信推送。

## 工作原理

```
每天 08:00 北京时间
  └─ GitHub Actions 触发
       ├─ LLM 扩展搜索词        (DeepSeek API)
       ├─ 多源抓取文献           (arXiv / Crossref / Semantic Scholar / GitHub / RSS)
       ├─ LLM 筛选 + 中文摘要   (DeepSeek API)
       ├─ LLM 热点聚类           (DeepSeek API)
       ├─ 渲染 Markdown + HTML  Digest
       └─ Server酱 → 微信推送 ✅
```

## 快速开始

### 1. Fork 此仓库 / 创建新仓库

```bash
git clone https://github.com/YOUR_USERNAME/auto-paper-collecter-cloud.git
cd auto-paper-collecter-cloud
```

### 2. 配置 GitHub Secrets

在仓库 Settings → Secrets and variables → Actions → New repository secret：

| Secret | 说明 | 必填 |
|--------|------|------|
| `AI_BASE_URL` | LLM API 地址 | ✅ |
| `AI_API_KEY` | LLM API Key | ✅ |
| `AI_MODEL` | 模型名称，如 `deepseek-chat` | ✅ |
| `SERVERCHAN_KEY` | Server酱 SendKey | ✅（微信推送） |
| `SEMANTIC_SCHOLAR_KEY` | S2 API Key（可选，提升速率） | ❌ |
| `GITHUB_TOKEN` | 已内置，无需手动添加 | - |

### 3. 自定义关键词

编辑 `state/config.json`：

```json
{
  "keywords": ["Quantum Gases", "Superconductivity", "Strongly Correlated Electrons"],
  "domain": "physics",
  "lookback_days": 5,
  "max_per_source": 12
}
```

### 4. 启用 Actions

推送后 GitHub Actions 自动开始按计划运行。也可以手动触发测试：
- Actions → Daily Paper Digest → Run workflow

## 本地运行

```bash
export AI_BASE_URL="https://api.deepseek.com/v1"
export AI_API_KEY="sk-xxx"
export AI_MODEL="deepseek-chat"
export SERVERCHAN_KEY="SCTxxx"

python pipeline.py              # 完整管线
python pipeline.py --no-llm     # 跳过 AI（仅原始结果）
python pipeline.py --fetch-only # 仅抓取不渲染
```

## 成本

- GitHub Actions：免费（每月 2000 分钟，每次运行 <3 分钟）
- DeepSeek API：每次约 ¥0.02-0.05（取决于文献数量）
- Server酱：免费版够用

## 项目结构

```
├── .github/workflows/daily.yml   # GitHub Actions 定时任务
├── scripts/
│   ├── common.py                 # 公共工具
│   ├── fetch.py                  # 多源文献抓取
│   ├── llm.py                    # AI 调用（查询扩展、筛选、摘要）
│   ├── render.py                 # 渲染 Markdown + HTML Digest
│   └── notify.py                 # 推送（Server酱 / 企业微信 / Telegram / Email）
├── pipeline.py                   # 主编排脚本
├── state/
│   ├── config.json               # 关键词 & 数据源配置
│   └── seen.json                 # 去重历史（自动更新）
└── digests/                      # 生成的日报
    ├── YYYY-MM-DD.md
    └── YYYY-MM-DD.html
```
