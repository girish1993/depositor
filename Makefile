.PHONY:install train api build-trainer build-api build-both down


#local only#
install:
	@python -m pip install -U pip
	@pip install -e .

train:
	@python -m src.modeling.trainer

api:
	@UVICORN_CMD="uvicorn api.main:app --reload --port 8000"; \
	MODEL_PATH=artifacts/models/latest_model.joblib \
	LABEL_ENCODER_PATH=artifacts/models/latest_label_encoder.joblib \
	CONFIG_PATH=configs/train.yaml \
	$$UVICORN_CMD

#----Docker-----#
#Build and run trainer
build-trainer:
	docker compose build trainer
	docker compose up trainer

#Build and run API
build-api:
	docker compose build api
	docker compose up api

# Build and run both trainer + API
build-both:
	docker compose build
	docker compose up

# down both
down:
	docker compose down
