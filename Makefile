.PHONY: test demo demo-llm effect docker-demo

test:            ## прогнать все тесты (190 штук, ~2 секунды)
	python -m pytest

demo:            ## end-to-end демо: 4 сценария + проверка аудит-лога
	python demo.py

demo-llm:        ## то же демо, но черновики генерирует локальная LLM (нужна запущенная Ollama)
	python demo.py --llm ollama:llama3.1:8b

effect:          ## пересчитать таблицы эффекта для docs/product.md
	python tools/effect_model.py

docker-demo:     ## то же самое в контейнере
	docker build -t support-ai-triage . && docker run --rm support-ai-triage
