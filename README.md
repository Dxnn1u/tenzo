# tenzo

從零開始實作的 tensor library，以 PyTorch 為參考目標。

目前處於早期學習階段，依循《Deep Learning from Scratch 3》（deZero）逐步實作自動微分，
設計上重視可維護性、可擴展性與穩定性。

> ⚠️ 開發中（0.x），API 可能隨時變動。

## 專案結構

```
docs/        # 相關文件
roadmap/     # 目前進度
src/tenzo/
├── core/    # Variable、Function、反向傳播
└── ops/     # 具體運算（Square、Exp…）
tests/       # 測試，結構鏡像 src/tenzo
```

依賴方向由上往下：`ops` → `core`。後續會依進度加入 `nn`、`optim` 與 `backend`。

## 開發

需要 [uv](https://docs.astral.sh/uv/) 與 Python 3.11+。

```bash
uv sync                  # 安裝依賴
uv run pytest            # 測試
uv run ruff check .      # Lint
uv run mypy src          # 型別檢查
```

## 授權

見 [LICENSE](LICENSE)。
