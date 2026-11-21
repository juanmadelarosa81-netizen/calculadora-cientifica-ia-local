# Calculadora Científica IA Local

Calculadora científica ejecutada en local con arquitectura de doble agente y verificación determinista mediante SymPy.

## Flujo
1. Agente 1 (Descripción): lectura sintáctica sin cálculo.
2. Formalización (Enviar): traducción a LaTeX limpio.
3. Agente 2 + SymPy (Solucionar): cálculo simbólico exacto, explicación paso a paso y registro en notas/.

## Uso
pip install -r requirements.txt
cp .env.example .env
python app.py
