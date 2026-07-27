.PHONY: test demo effect docker-demo

test:            ## прогнать все тесты (145 штук, ~2 секунды)
	python -m pytest

demo:            ## end-to-end демо: 4 сценария + проверка аудит-лога
	python demo.py

effect:          ## пересчитать таблицы эффекта для docs/product.md
	python tools/effect_model.py

docker-demo:     ## то же самое в контейнере
	docker build -t support-ai-triage . && docker run --rm support-ai-triage
