# Variáveis
PYTHON = python3
MAIN_SCRIPT = main.py
DATA_FILE = filmes.csv
SIMILARITY_THRESHOLD = 0.5
RATING_THRESHOLD = 0
REVENUE_THRESHOLD = 300 # milhões (parâmetros usados: 0, 100, 200, 300)

# Alvos
.PHONY: all run clean

# Alvo padrão
all: run

# Executar o script principal
run:
	$(PYTHON) $(MAIN_SCRIPT) --file_path $(DATA_FILE) --similarity_threshold $(SIMILARITY_THRESHOLD) --rating_threshold $(RATING_THRESHOLD) --revenue_threshold $(REVENUE_THRESHOLD)

# Limpar arquivos temporários (se houver)
clean:
	rm -f *.pyc
	rm -rf __pycache__