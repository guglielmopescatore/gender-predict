# Makefile per Gender Prediction API
# Esegui dalla directory api/

.PHONY: help deploy test verify logs stop clean status all

# Default
help:
	@echo "Gender Prediction API - Deployment Commands"
	@echo "=========================================="
	@echo "  make deploy   - Deploy su Modal"
	@echo "  make test     - Test del deployment"
	@echo "  make verify   - Verifica versione modello"
	@echo "  make logs     - Mostra logs"
	@echo "  make status   - Status dell'app"
	@echo "  make stop     - Ferma l'app"
	@echo "  make clean    - Pulisci deployment"
	@echo "  make all      - Deploy completo + test"

# Deploy
deploy:
	@echo "🚀 Deploying to Modal..."
	@modal deploy modal_deployment.py

# Test deployment
test:
	@echo "🧪 Testing deployment..."
	@modal run modal_deployment.py::test_model_files
	@modal run modal_deployment.py::test_academic_features

# Verifica modello
verify:
	@echo "🔍 Verifying model version..."
	@python verify_model_version.py

# Logs
logs:
	@modal app logs gender-prediction-academic

# Status
status:
	@echo "📊 App status:"
	@modal app list | grep gender-prediction-academic || echo "App not found"

# Stop
stop:
	@echo "🛑 Stopping app..."
	@modal app stop gender-prediction-academic

# Clean
clean: stop
	@echo "🧹 Cleaning up..."
	@-modal volume rm gender-prediction-models 2>/dev/null || true

# Deploy completo
all: deploy test
	@echo ""
	@echo "✅ Deployment completed!"
	@APP_URL=$$(modal app list | grep "gender-prediction-academic" | grep -oE 'https://[^ ]+' | head -1); \
	if [ -n "$$APP_URL" ]; then \
		echo "🌐 App URL: $$APP_URL"; \
		echo "📚 API Docs: $$APP_URL/docs"; \
		echo ""; \
		echo "Test with:"; \
		echo "  curl -X POST \"$$APP_URL/predict\" \\"; \
		echo "    -H \"Content-Type: application/json\" \\"; \
		echo "    -d '{\"names\": \"Mario Rossi\"}'"; \
	fi