# DeepSearch 检索链路说明（S2）

本文说明 Lumenarium 的 S2 资产检索如何连到 Omniverse DeepSearch，以及认证、代理、
ngrok 隧道、验证的完整过程。**范围边界**：本文覆盖 Python 侧（本仓库内）能确定的
全部内容；UE 侧透明代理与 ngrok 隧道本身的启动步骤不在本仓库，见文末「盲区」。

## 1. 概览

DeepSearch 是 Calvin Gu 与导师在腾讯的团队提供的生产级资产检索服务，索引
十万量级资产、单次查询亚秒内返回最相关候选。它在管线中对应 S2 阶段，上游是 S1
（场景解析产出实例掩码与类别标签），下游是 S3（位姿推断）。

链路（文字图）：

```
A10 上的 pipeline (modules/retrieval.py)
   │  POST {description, image_similarity_search, limit, search_path?}
   ▼
OMNIVERSE_DEEPSEARCH_URL
   ├─ 生产：https://ov.qq.com/search（直连）
   └─ 测试：https://miller-unshapeable-melany.ngrok-free.dev/search（ngrok 公网隧道）
               │
               ▼
         UE 透明代理（注入 Omniverse 凭证）──→ Omniverse API
```

## 2. 开启开关

```bash
export IMAGINARIUM_USE_DEEPSEARCH=1
```

设为 `1` 时，`modules/retrieval.py:547` 走 DeepSearch 分支；否则走旧的本地检索
（legacy）。这是 v4-deepsearch 与之前版本在 S2 的唯一分叉点。

## 3. 请求契约（`modules/retrieval.py:386` `_deepsearch_search`）

对每个被检测物体发一个 `POST`，body 字段：

```json
{
  "limit": 10,
  "search_path": "<可选，路径作用域>",
  "image_similarity_search": ["<掩码图像 base64>"],
  "description": "<类别文本标签>"
}
```

- `description` 来自 S1 的类别标签；
- `image_similarity_search` 是实例掩码的 base64 编码（视觉相似检索）；
- `search_path` 默认留空（`OMNIVERSE_DEEPSEARCH_SEARCH_PATH`），因为 API/代理没有
  UE「当前文件夹」上下文，部署方可能把同一集合存在别处。

返回结果把候选 url 映射到资产 CSV（`asset_data/imaginarium_asset_info.csv`）的
`name_en`。

## 4. 认证：两种方式

### 4.1 UE 透明代理注入（生产默认）

UE 侧透明代理自己注入 Omniverse 凭证。这种情况下 Python 侧**不需要**提供 token，
`OMNIVERSE_JWT_TOKEN` 留空即可。

### 4.2 客户端 Basic 认证（代理转发客户端认证时）

当代理被配置为转发客户端认证时，Python 侧提供 JWT：

```python
auth = ("$omni-api-token", jwt_token)   # requests 把它转成 Basic
```

对应 `Authorization: Basic base64("$omni-api-token:<JWT>")`。

## 5. 本地认证代理（`tools/deepsearch_proxy.py`）

当本机没有 UE 代理、但持有 JWT 时，用这个 Python 代理把本地请求加认证后转发到
Omniverse：

```bash
export OMNIVERSE_DEEPSEARCH_BASE="https://ov.qq.com"   # 默认
export OMNIVERSE_PROXY_PORT="9192"                      # 默认
export OMNIVERSE_JWT_TOKEN="<三段式 JWT>"
# 或：export OMNIVERSE_JWT_TOKEN_FILE="/path/to/omni.jwt"  # 读取后自动删除
python tools/deepsearch_proxy.py
```

代理逻辑：
- 在 `:9192` 监听，`GET`/`POST` 均转发到 `OMNIVERSE_DEEPSEARCH_BASE + path`
- 注入 `Authorization: Basic base64("$omni-api-token:<JWT>")`
- JWT 必须是三段式（`.split(".")` 长度为 3），否则启动即报错
- `OMNIVERSE_JWT_TOKEN_FILE` 读完后会 `os.remove` 删除，避免落盘

若走本代理，把 `OMNIVERSE_DEEPSEARCH_URL` 指向 `http://<host>:9192/search` 即可。

## 6. 环境变量清单（`.env.lumenarium.example`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNIVERSE_DEEPSEARCH_URL` | `https://ov.qq.com/search` | S2 检索端点 |
| `OMNIVERSE_DEEPSEARCH_SEARCH_PATH` | 空 | 可选路径作用域 |
| `OMNIVERSE_JWT_TOKEN` | 空 | 仅端点本身要 JWT 时填 |
| `OMNIVERSE_DEEPSEARCH_MAX_ATTEMPTS` | 6 | 重试次数 |
| `OMNIVERSE_DEEPSEARCH_TIMEOUT` | 120 | 单次超时（秒） |
| `OMNIVERSE_DEEPSEARCH_RETRY_DELAY` | 2 | 重试退避 |
| `OMNIVERSE_DEEPSEARCH_WORKERS` | 4（API）/ 见脚本 | S2 并发数 |
| `SCENEPROOF_API_DEEPSEARCH_WORKERS` | 4 | API 服务内的并发 |

重试策略：429 / 5xx 指数退避，最多 `MAX_ATTEMPTS` 次。

## 7. 连接验证（`A10_TESTING.md`）

先测连通性（测试环境的 ngrok 公网 URL）：

```bash
curl --fail --show-error \
  "https://miller-unshapeable-melany.ngrok-free.dev/search?description=house&limit=2"
```

再跑管线并核 S2 日志必须出现：

```
Running retrieval with Omniverse DeepSearch...
```

且 `retrieval_results.json` 与 `retrieval_results_final.json` 一致，每个非背景物体的
`data[obj_name][0][0]` 是资产名字符串。

## 8. S2 并发 smoke（`scripts/run_deepsearch_s2_concurrency_smoke5.sh`）

对比 1 worker 与 8 worker 的耗时与成功率，验证生产并发不破坏结果：

```bash
DEEPSEARCH_S2_MANIFEST=/tmp/sceneba_repair_smoke5.txt \
DEEPSEARCH_S2_RESULTS_ROOT=a10_reusable_results/paper30 \
DEEPSEARCH_S2_SOURCE_VERSION=v4_deepsearch \
bash scripts/run_deepsearch_s2_concurrency_smoke5.sh
```

脚本复用冻结的 S0/S1 缓存，只重跑 S2（`IMAGINARIUM_STOP_AFTER_STAGE=S2`），
落盘 `runtime_w{1,8}.jsonl` 与 `summary.json`。

## 9. 盲区（需向 Calvin Gu / 导师团队索取）

以下内容**不在本仓库**，复现时需向团队获取：

1. **UE 侧透明代理的启动方式**：UE 工程、代理插件、如何指向 Omniverse Nucleus /
   资产索引。本仓库只有 Python 侧如何连到它。
2. **ngrok 隧道的建立**：隧道命令、`ngrok-free.dev` 域名如何绑定到 UE 代理端口、
   隧道是否长期稳定还是按次启动。
3. **JWT 的签发与有效期**：`OMNIVERSE_JWT_TOKEN` 从哪申请、多久过期。

拿到这三项后，配上第 6 节的 env，即可在本仓库内完整跑通 S2 检索。
