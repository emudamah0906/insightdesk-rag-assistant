.PHONY: demo chat eval eda classify api ui
demo:     ## ask the RAG assistant a sample question
	python3 src/rag.py "How long do refunds take and how do I get one?"
chat:     ## interactive RAG REPL
	python3 src/rag.py
eval:     ## run the golden-set evaluation
	python3 src/eval.py
eda:      ## exploratory data analysis on tickets
	python3 scripts/01_eda.py
classify: ## train the PyTorch intent classifier (needs: pip install torch)
	python3 src/classifier_torch.py
api:      ## serve the API (needs: pip install fastapi uvicorn pydantic)
	uvicorn src.api:app --reload --port 8080
ui:       ## launch the Streamlit chat UI (needs: pip install streamlit)
	streamlit run src/app.py
