.PHONY: help install pipeline backend frontend up down logs clean

help:
	@echo "Belfast Sentinel — common tasks"
	@echo "  make install    install Python + Node deps"
	@echo "  make pipeline   build grid, fetch data, train model"
	@echo "  make backend    run FastAPI dev server (port 8000)"
	@echo "  make frontend   run React dev server (port 3000)"
	@echo "  make up         docker compose up --build"
	@echo "  make down       docker compose down"
	@echo "  make clean      remove generated outputs"

install:
	pip install -r data_preparation/requirements.txt
	pip install -r backend/requirements.txt
	cd frontend && npm install

pipeline:
	cd data_preparation && python run_pipeline.py

backend:
	cd backend && uvicorn api:app --reload --port 8000

frontend:
	cd frontend && npm start

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf data_preparation/outputs/*
	rm -f backend/belfast_sentinel_model.* backend/belfast_grid_with_features.geojson
