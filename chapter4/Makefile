# Docker CLIの絶対パス（Docker Desktop標準の場所。別の場所の場合は make start.engine DOCKER_BIN=/path/to/docker）
DOCKER_BIN ?= /Applications/Docker.app/Contents/Resources/bin/docker
# イメージ取得時に docker-credential-desktop が必要なので、同じ bin を PATH に追加
DOCKER_BIN_DIR := /Applications/Docker.app/Contents/Resources/bin

.PHONY: start.engine
start.engine:
	@PATH="$(DOCKER_BIN_DIR):$$PATH" $(DOCKER_BIN) compose up -d

.PHONY: stop.engine
stop.engine:
	@PATH="$(DOCKER_BIN_DIR):$$PATH" $(DOCKER_BIN) compose down

.PHONY: create.index
create.index:
	@uv run python -m src.scripts.create_index

.PHONY: delete.index
delete.index:
	@uv run python -m src.scripts.delete_index

# .env が読めているか・Settings が使えるかを確認（ターミナルは chapter4 で実行すること）
.PHONY: check.env
check.env:
	@uv run python -c "import src.configs as c; print('env_path:', c._env_path, 'exists:', c._env_path.is_file()); from src.configs import Settings; s = Settings(); print('OK model:', s.openai_model)"

# docker の場所を確認する（DOCKER_BIN が違う場合の調査用）
.PHONY: docker-check
docker-check:
	@echo "DOCKER_BIN=$(DOCKER_BIN)"; ls -la /Applications/Docker.app/Contents/Resources/bin/ 2>/dev/null || echo "※ 上記パスに docker が無い場合は、Docker Desktop の設定で CLI のパスを確認してください"