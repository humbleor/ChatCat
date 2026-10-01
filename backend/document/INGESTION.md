# 文档版本化入库

当前流程为：文件落盘 → 解析与分块 → 父块幂等入库 → 叶子块向量化幂等入库 → 校验与发布 → 旧版下线与后台软删除。

这些阶段有先后依赖，每个 worker 内顺序执行。HTTP 上传返回任务 ID 后，工作由后台线程执行；每个服务进程一个消费者，不代表同一文档的六个阶段并行，也没有吞吐提升的测量结论。

## 数据与发布

- 每次上传生成独立 document_id，文件写入 data/documents/<document_id>/<filename>。先写 .part，再 fsync 和原子 rename；完整源文件落盘后，在同一 PostgreSQL 事务中创建 staging 版本和 document_jobs 队列记录。
- document_heads 以文件名标识逻辑文档，保存 current_document_id 和单调版本计数。文件名相同视为更新，不是按内容去重。
- 解析结果保存为版本目录内 chunks.json。重试复用该结果，避免解析规则变化导致 chunk_id 不稳定。分块 ID 已包含 document_id，父块按 chunk_id upsert，Milvus 以稳定 chunk_uid 为主键 full upsert，BM25 由服务端计算。
- 发布前校验父子关系、父块数量、父块 ID 集合、Milvus Strong 读取的叶子块 ID 集合。随后锁定 document_heads 行，在一个 PostgreSQL 事务内更新指针、新版 active、旧版 retiring。
- 检索使用一次读取的当前版本 ID 过滤，同时过滤 is_deleted。数据库不可用时返回检索失败，不回退到全部版本。未纳管的历史 document_id 为空的数据在首次新版发布前继续可见。
- 已发布的较新版本不会被较晚完成的较旧上传覆盖。较新上传失败时，可以继续使用此前成功版本。
- 发布后旧版立即从新检索的过滤条件中消失，默认延迟 60 秒才软删除。这个间隔缓解正在执行的旧版检索与清理冲突，但不是无限期读事务保护。

## 任务与故障

- 队列和阶段进度都存入 PostgreSQL；启动时消费者扫描 pending/running 任务。PG session advisory lock 保证多个进程不会同时领取同一任务，进程退出后锁由数据库释放。
- 重试为至少一次执行。解析复用 manifest，父块和叶子块重复写入保持幂等。不是跨 PostgreSQL / Milvus 的分布式事务。
- 默认失败后等待 10 秒、最多尝试 3 次；无有效内容、源文件丢失等确定性错误直接失败。staging 失败不会改变当前指针；用户可以重新上传。
- 若发布已经提交但进度记录尚未完成就宕机，重启后识别已发布状态，只完成收尾，不重新把 active 版本写回 staging。
- retiring / soft_deleting 记录构成独立的持久化清理队列；清理失败保持可重试，后台继续扫描。删除、恢复、清理、物理清除共享版本操作锁；恢复时先校验分块，再切换指针。
- 文件上传中断无法恢复未传完的字节，需客户端重新上传。.part 会在正常异常处理中移除；硬宕机可能留下未纳管的 .part 或源文件，但它们不会进入检索。
- 旧同步 POST /documents/upload 返回 409，统一使用 /documents/upload/async，避免绕过 staging 发布。

## 启动与配置

仍按原命令启动 Docker 基础服务与后端。worker 由后端 lifespan 自动启动，不需要新增终端或独立队列服务。已有数据库启动时执行幂等迁移；保留既有 active 版本并建立当前指针。前端修改后运行 npm run build。

配置见 .env.example：UPLOAD_MAX_ATTEMPTS、UPLOAD_RETRY_DELAY_SECONDS、DOCUMENT_RETIRE_GRACE_SECONDS。

运行无需模型下载的回归测试：

```bash
.venv/bin/pytest -q tests/test_document_ingestion.py
```

默认使用临时 SQLite，仅跳过 PostgreSQL 并发语义和真实 Milvus 测试。完整验证可设置 TEST_DOCUMENT_DATABASE_URL 指向专用测试数据库、TEST_DOCUMENT_MILVUS_URI 指向隔离 Milvus。测试创建并删除测试表；不要指向业务数据库。Milvus 测试仅使用随机命名的 ingestion_test_ 集合。
