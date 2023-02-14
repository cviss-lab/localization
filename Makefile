all: serve

serve: $(WEB_ASSETS)
	flask --app localization_server run

serve-all: $(WEB_ASSETS)
	flask --app localization_server run --host 0.0.0.0

deps: pull-sample
	pip3 install -r requirements.txt

.PHONY: all serve serve-all deps