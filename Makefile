# Nome do ambiente virtual
VENV = .venv

# Variável para comandos do Python no ambiente virtual
PYTHON = $(VENV)/bin/python

# Configurar o ambiente virtual e instalar dependências
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

# Rodar o programa principal
run: $(VENV)/bin/activate
	$(PYTHON) src/main.py

# Instalar dependências do projeto
install: $(VENV)/bin/activate
	$(VENV)/bin/pip install -r requirements.txt

# Limpar arquivos gerados
clean:
	rm -rf __pycache__ $(VENV)

# Atualizar requirements.txt
freeze: $(VENV)/bin/activate
	$(VENV)/bin/pip freeze > requirements.txt

# Exibir ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make setup      - Configurar o ambiente virtual e instalar dependências"
	@echo "  make run        - Executar o programa principal"
	@echo "  make install    - Instalar as dependências do projeto"
	@echo "  make clean      - Limpar arquivos temporários"
	@echo "  make freeze     - Atualizar o arquivo requirements.txt"
