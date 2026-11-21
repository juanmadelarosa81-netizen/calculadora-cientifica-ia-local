import json
import os
import re
from datetime import datetime
import requests
from flask import Flask, jsonify, render_template_string, request

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_DISPONIBLE = True
except Exception:
    SYMPY_DISPONIBLE = False

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTAS_DIR = os.path.join(BASE_DIR, "notas")
os.makedirs(NOTAS_DIR, exist_ok=True)


def cargar_env_local():
    ruta_env = os.path.join(BASE_DIR, ".env")
    if os.path.isfile(ruta_env):
        with open(ruta_env, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith("#") and "=" in linea:
                    clave, valor = linea.split("=", 1)
                    os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


cargar_env_local()

# Motor local: FreeLLMAPI (puerto 3001)
FREELLM_URL = os.getenv("FREELLM_URL", "http://127.0.0.1:3001/v1/chat/completions")
FREELLM_API_KEY = os.getenv("FREELLM_API_KEY", "")
FREELLM_MODEL = os.getenv("FREELLM_MODEL", "auto")

historial_sesiones = []


def limpiar_salida_texto(texto: str) -> str:
    t = texto.replace("```latex", "").replace("```", "")
    t = t.replace("$$", "").replace("$", "")
    t = t.replace("**", "").replace("### ", "").replace("## ", "").replace("# ", "")
    t = t.replace("\\left(", "(").replace("\\right)", ")")
    t = t.replace("\\cdot", "·").replace("\\times", "×").replace("\\div", "÷")
    t = t.replace("\\pi", "π").replace("\\sqrt", "√")
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1 / \2)", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalizar_para_sympy(expr: str) -> str:
    s = expr.strip()
    s = s.replace("$$", "").replace("$", "")
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*")
    s = s.replace("\\div", "/").replace("÷", "/")
    s = s.replace("\\pi", "pi").replace("π", "pi")
    s = s.replace("√(", "sqrt(")
    s = re.sub(r"√(\d+(\.\d+)?)", r"sqrt(\1)", s)
    s = s.replace("²", "**2").replace("³", "**3")

    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    for fn in ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "ln", "log", "exp"]:
        s = s.replace(f"\\{fn}", fn)

    s = re.sub(r"\bln\(", "log_e(", s)
    s = re.sub(r"\blog\(", "log10(", s)
    s = s.replace("log_e(", "log(")
    s = s.replace("{", "(").replace("}", ")")
    return s


def calcular_exacto_sympy(formula_orig: str, latex_expr: str) -> str:
    if not SYMPY_DISPONIBLE:
        return ""

    transformaciones = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    contexto = {
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "I": sp.I,
        "sqrt": sp.sqrt,
        "cbrt": sp.cbrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "log": sp.log,
        "log10": lambda x: sp.log(x, 10),
        "exp": sp.exp,
        "abs": sp.Abs,
        "fact": sp.factorial,
    }

    for candidata in [formula_orig, latex_expr]:
        if not candidata:
            continue
        try:
            limpia = normalizar_para_sympy(candidata)
            if "=" in limpia and "==" not in limpia and "<=" not in limpia and ">=" not in limpia and "!=" not in limpia:
                izq, der = limpia.split("=", 1)
                e1 = parse_expr(izq, local_dict=contexto, transformations=transformaciones)
                e2 = parse_expr(der, local_dict=contexto, transformations=transformaciones)
                sols = sp.solve(sp.Eq(e1, e2))
                if sols:
                    return ", ".join(str(sp.simplify(sol)) for sol in sols)
            else:
                expresion = parse_expr(limpia, local_dict=contexto, transformations=transformaciones)
                simplificada = sp.simplify(expresion)
                if simplificada.is_number:
                    if simplificada.is_Integer:
                        return str(int(simplificada))
                    val_num = sp.N(simplificada, 10)
                    texto_num = f"{float(val_num):.10g}" if val_num.is_real else str(simplificada)
                    if str(simplificada) != texto_num and not simplificada.is_Rational:
                        return f"{simplificada} (≈ {texto_num})"
                    return texto_num
                return str(simplificada)
        except Exception:
            continue

    return ""


def llamar_motor_ia(prompt: str, fallback_texto: str) -> str:
    headers = {"Content-Type": "application/json"}
    if FREELLM_API_KEY:
        headers["Authorization"] = f"Bearer {FREELLM_API_KEY}"

    try:
        resp = requests.post(
            FREELLM_URL,
            headers=headers,
            json={
                "model": FREELLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un profesor y motor matemático exacto. "
                            "Responde siempre en español claro y directo, sin pensar en voz alta."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                texto = (choices[0].get("message", {}).get("content") or "").strip()
                if texto:
                    return texto
    except Exception:
        pass

    return fallback_texto


def guardar_auditoria(registro: dict) -> str:
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"calc_{marca}.txt"
    ruta = os.path.join(NOTAS_DIR, nombre)
    contenido = (
        f"FECHA: {registro.get('fecha', '')}\n"
        f"FORMULA ORIGINAL: {registro.get('formula_original', '')}\n"
        f"DESCRIPCION SINTACTICA: {registro.get('descripcion_sintactica', '')}\n"
        f"LATEX AGENTE 1: {registro.get('latex_agente_1', '')}\n"
        f"RESOLUCION Y PASOS:\n{registro.get('resolucion', '')}\n"
    )
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    return nombre


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calculadora Científica IA Local</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: monospace;
    background: #121212;
    color: #e0e0e0;
    margin: 0;
    padding: 14px;
  }
  .app-container {
    max-width: 780px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    background: #181818;
    border: 1px solid #2b2b2b;
    border-radius: 8px;
    padding: 10px 14px;
  }
  .app-title {
    margin: 0;
    font-size: 1.05rem;
    color: #ffffff;
  }
  .web-calc-link {
    color: #74c0fc;
    text-decoration: none;
    font-weight: bold;
    font-size: 0.82rem;
    border: 1px solid #1864ab;
    background: #132238;
    padding: 6px 10px;
    border-radius: 5px;
  }
  .web-calc-link:hover {
    background: #1864ab;
    color: #fff;
  }
  .panel {
    background: #161616;
    border: 1px solid #2b2b2b;
    border-radius: 8px;
    padding: 12px;
  }
  .panel-label {
    color: #4dabf7;
    font-weight: bold;
    font-size: 0.88rem;
    margin-bottom: 8px;
    display: block;
  }
  .panel-label.agente2 {
    color: #69db7c;
  }
  .panel-label.inferior {
    color: #ffa94d;
  }
  .row-box {
    display: flex;
    gap: 8px;
    align-items: stretch;
  }
  .scroll-box {
    flex: 1;
    min-height: 92px;
    max-height: 170px;
    overflow-y: auto;
    background: #1c1c1c;
    border: 1px solid #383838;
    color: #f1f3f5;
    font-family: monospace;
    font-size: 0.9rem;
    padding: 9px;
    border-radius: 6px;
    resize: vertical;
    line-height: 1.4;
  }
  .btn-col {
    display: flex;
    flex-direction: column;
    gap: 5px;
    width: 104px;
  }
  .btn-clear {
    background: #232323;
    color: #ff8787;
    border: 1px solid #495057;
    border-radius: 5px;
    padding: 5px;
    font-weight: bold;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .btn-clear:hover {
    background: #2f2f2f;
  }
  .btn-desc {
    background: #5f3dc4;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 7px 6px;
    font-weight: bold;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .btn-desc:hover {
    background: #7048e8;
  }
  .btn-primary {
    background: #1864ab;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 7px 6px;
    font-weight: bold;
    cursor: pointer;
    font-size: 0.82rem;
    flex: 1;
  }
  .btn-primary:hover {
    background: #1c7ed6;
  }
  .btn-solve {
    background: #2b8a3e;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 10px 8px;
    font-weight: bold;
    cursor: pointer;
    font-size: 0.84rem;
    flex: 1;
  }
  .btn-solve:hover {
    background: #2f9e44;
  }
  .desc-sintaxis-box {
    display: none;
    margin-top: 8px;
    background: #1a1829;
    border: 1px solid #5f3dc4;
    border-radius: 6px;
    padding: 8px 10px;
    color: #d0bfff;
    font-size: 0.82rem;
    white-space: pre-wrap;
    line-height: 1.4;
    max-height: 130px;
    overflow-y: auto;
  }
  .keypad-box {
    background: #181818;
    border: 1px solid #2b8a3e;
    border-radius: 8px;
    padding: 8px;
  }
  .keypad-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
  }
  .key-btn {
    background: #222;
    color: #fff;
    border: 1px solid #343a40;
    border-radius: 5px;
    padding: 9px 4px;
    text-align: center;
    font-size: 0.86rem;
    font-weight: bold;
    cursor: pointer;
    user-select: none;
  }
  .key-btn:hover {
    background: #2c2c2c;
    border-color: #69db7c;
  }
  .keypad-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    padding: 5px 8px;
    background: #111;
    border-radius: 5px;
    border: 1px solid #252525;
  }
  .nav-btn {
    background: #222;
    color: #69db7c;
    border: 1px solid #343a40;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: bold;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .nav-btn:hover {
    background: #2c2c2c;
  }
  .pag-info {
    color: #adb5bd;
    font-size: 0.84rem;
    font-weight: bold;
  }
  .resultado-scroll {
    min-height: 150px;
    max-height: 360px;
    overflow-y: auto;
    background: #111;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .entrada-historial {
    background: #181818;
    border-left: 3px solid #ffa94d;
    border-radius: 5px;
    padding: 10px;
  }
  .meta-linea {
    font-size: 0.78rem;
    color: #adb5bd;
    margin-bottom: 6px;
  }
  .formula-orig {
    color: #ffd43b;
    font-weight: bold;
    font-size: 0.92rem;
    margin-bottom: 8px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .pasos-texto {
    color: #e9ecef;
    font-size: 0.86rem;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.45;
  }
  .estado-msg {
    font-size: 0.78rem;
    color: #74c0fc;
    margin-top: 6px;
    min-height: 18px;
  }
</style>
</head>
<body>
<div class="app-container">
  <div class="top-bar">
    <h1 class="app-title">Calculadora Científica IA Local</h1>
    <a class="web-calc-link" href="https://calculator.com/es/calculate/scientific" target="_blank" rel="noopener noreferrer">
      Calculadora Científica Web
    </a>
  </div>

  <!-- 1. AGENTE PROBLEMA ORIGINAL -->
  <div class="panel">
    <label class="panel-label" for="agente_problema_original">
      Agente problema original (escribe, pega o usa el teclado)
    </label>
    <div class="row-box">
      <textarea
        id="agente_problema_original"
        class="scroll-box"
        placeholder="Escribe o pega aquí la ecuación con símbolos científicos..."
      ></textarea>
      <div class="btn-col">
        <button type="button" class="btn-clear" onclick="limpiarCaja('agente_problema_original')">[X]</button>
        <button type="button" class="btn-desc" onclick="analizarSintaxisOriginal()">Descripción</button>
        <button type="button" class="btn-primary" onclick="enviarProblemaOriginal()">Enviar</button>
      </div>
    </div>
    <div id="caja_descripcion_sintaxis" class="desc-sintaxis-box"></div>
    <div id="estado_agente_1" class="estado-msg"></div>
  </div>

  <!-- TECLADO CIENTÍFICO PAGINADO (10 PÁGINAS) -->
  <div class="keypad-box">
    <div id="keypad_grid" class="keypad-grid"></div>
    <div class="keypad-nav">
      <button type="button" class="nav-btn" onclick="cambiarPagina(-1)">&#9668; Ant</button>
      <span id="label_pagina" class="pag-info">Pág 1/10</span>
      <button type="button" class="nav-btn" onclick="cambiarPagina(1)">Sig &#9654;</button>
    </div>
  </div>

  <!-- 2. AGENTE RESOLUCIÓN Y ACLARACIÓN -->
  <div class="panel">
    <label class="panel-label agente2" for="agente_resolucion_aclaracion">
      Agente resolución y aclaración (LaTeX recibido para resolver)
    </label>
    <div class="row-box">
      <textarea
        id="agente_resolucion_aclaracion"
        class="scroll-box"
        placeholder="Aquí llegará la fórmula en LaTeX legible desde el Agente problema original..."
      ></textarea>
      <div class="btn-col">
        <button type="button" class="btn-clear" onclick="limpiarCaja('agente_resolucion_aclaracion')">[X]</button>
        <button type="button" class="btn-solve" onclick="solucionarLatex()">Solucionar</button>
      </div>
    </div>
    <div id="estado_agente_2" class="estado-msg"></div>
  </div>

  <!-- 3. CUADRÍCULA INFERIOR: FÓRMULA ORIGINAL CON SÍMBOLOS CIENTÍFICOS + EXPLICACIÓN PASO A PASO -->
  <div class="panel">
    <label class="panel-label inferior">
      Resultado y desglose paso a paso (fórmula original + explicación)
    </label>
    <div id="cuadricula_inferior" class="resultado-scroll">
      {% for item in historial %}
        <div class="entrada-historial">
          <div class="meta-linea">{{ item.fecha }} · Archivo: {{ item.archivo }}</div>
          <div class="formula-orig">Fórmula original: {{ item.formula_original }}</div>
          <div class="pasos-texto">{{ item.resolucion }}</div>
        </div>
      {% else %}
        <div id="sin_resultados" style="color: #777; font-size: 0.85rem;">
          Aún no hay cálculos resueltos. Envía una fórmula en el Agente problema original y pulsa Solucionar.
        </div>
      {% endfor %}
    </div>
  </div>
</div>

<script>
const paginasTeclado = [
  ['+', '-', '*', '/', '=', '(', ')', '.', '%', '±', '^', '√'],
  ['sin(', 'cos(', 'tan(', 'asin(', 'acos(', 'atan(', 'sinh(', 'cosh(', 'tanh(', 'rad', 'deg', 'π'],
  ['log(', 'ln(', 'exp(', 'abs(', 'fact(', 'mod', 'sqrt(', 'cbrt(', '1/x', 'x²', 'x³', 'x^y'],
  ['e', 'π', 'φ', '∞', 'i', 'θ', 'λ', 'μ', 'σ', 'Δ', 'Ω', 'Σ'],
  ['[', ']', '{', '}', 'det(', 'tr(', 'inv(', 'T', 'matrix(', 'row(', 'col(', 'I'],
  ['AND', 'OR', 'NOT', 'XOR', '==', '!=', '<', '>', '<=', '>=', '&&', '||'],
  ['d/dx', '∫', 'lim', 'd²', '∂', '∇', 'dx', 'dy', 'f(x)', 'g(x)', 'n!', 'P(n,r)'],
  ['∈', '∉', '⊂', '⊆', '∪', '∩', '∅', '∀', '∃', 'ℕ', 'ℤ', 'ℝ'],
  ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ'],
  ['ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
];

let paginaActual = 0;
let formulaOriginalActual = '';
let descripcionSintacticaActual = '';

function renderizarTeclado() {
  const grid = document.getElementById('keypad_grid');
  const label = document.getElementById('label_pagina');
  grid.innerHTML = '';
  paginasTeclado[paginaActual].forEach(sim => {
    const btn = document.createElement('div');
    btn.className = 'key-btn';
    btn.textContent = sim;
    btn.onclick = () => insertarEnOriginal(sim);
    grid.appendChild(btn);
  });
  label.textContent = `Pág ${paginaActual + 1}/${paginasTeclado.length}`;
}

function cambiarPagina(delta) {
  paginaActual = (paginaActual + delta + paginasTeclado.length) % paginasTeclado.length;
  renderizarTeclado();
}

function insertarEnOriginal(sim) {
  const el = document.getElementById('agente_problema_original');
  const start = el.selectionStart ?? el.value.length;
  const end = el.selectionEnd ?? el.value.length;
  el.value = el.value.slice(0, start) + sim + el.value.slice(end);
  el.focus();
  el.selectionStart = el.selectionEnd = start + sim.length;
}

function limpiarCaja(id) {
  const el = document.getElementById(id);
  el.value = '';
  if (id === 'agente_problema_original') {
    descripcionSintacticaActual = '';
    const descEl = document.getElementById('caja_descripcion_sintaxis');
    descEl.style.display = 'none';
    descEl.textContent = '';
    document.getElementById('estado_agente_1').textContent = '';
  }
  el.focus();
}

async function analizarSintaxisOriginal() {
  const origEl = document.getElementById('agente_problema_original');
  const descEl = document.getElementById('caja_descripcion_sintaxis');
  const estado1 = document.getElementById('estado_agente_1');
  const formula = origEl.value.trim();
  if (!formula) {
    estado1.textContent = 'Escribe o pega una fórmula primero.';
    return;
  }
  formulaOriginalActual = formula;
  estado1.textContent = 'Analizando sintaxis paso a paso sin calcular...';
  try {
    const res = await fetch('/api/descripcion_sintactica', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ formula })
    });
    const data = await res.json();
    descripcionSintacticaActual = data.descripcion || '';
    descEl.textContent = descripcionSintacticaActual;
    descEl.style.display = descripcionSintacticaActual ? 'block' : 'none';
    estado1.textContent = 'Lectura sintáctica lista. Revisa el desglose y pulsa Enviar.';
  } catch (e) {
    estado1.textContent = 'Error al generar la descripción sintáctica.';
  }
}

async function enviarProblemaOriginal() {
  const origEl = document.getElementById('agente_problema_original');
  const destEl = document.getElementById('agente_resolucion_aclaracion');
  const estado1 = document.getElementById('estado_agente_1');
  const formula = origEl.value.trim();
  if (!formula) {
    estado1.textContent = 'Escribe o pega una fórmula primero.';
    return;
  }
  if (formulaOriginalActual !== formula) {
    descripcionSintacticaActual = '';
  }
  formulaOriginalActual = formula;
  estado1.textContent = 'Convirtiendo a LaTeX desde la sintaxis para el Agente 2...';
  try {
    const res = await fetch('/api/a_latex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        formula,
        descripcion: descripcionSintacticaActual
      })
    });
    const data = await res.json();
    destEl.value = data.latex || formula;
    destEl.scrollTop = 0;
    estado1.textContent = 'Enviado como LaTeX al Agente resolución y aclaración.';
  } catch (e) {
    destEl.value = formula;
    estado1.textContent = 'Pasado directamente al Agente resolución y aclaración.';
  }
}

async function solucionarLatex() {
  const latexEl = document.getElementById('agente_resolucion_aclaracion');
  const origEl = document.getElementById('agente_problema_original');
  const estado2 = document.getElementById('estado_agente_2');
  const cuadricula = document.getElementById('cuadricula_inferior');
  const vacio = document.getElementById('sin_resultados');

  const latexTexto = latexEl.value.trim();
  const formulaOrig = (origEl.value.trim() || formulaOriginalActual || latexTexto);
  if (!latexTexto) {
    estado2.textContent = 'No hay expresión o LaTeX para solucionar.';
    return;
  }

  estado2.textContent = 'Resolviendo cálculo y preparando explicación paso a paso...';
  try {
    const res = await fetch('/api/solucionar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        formula_original: formulaOrig,
        descripcion_sintactica: descripcionSintacticaActual,
        latex: latexTexto
      })
    });
    const data = await res.json();
    if (vacio) vacio.remove();

    const bloque = document.createElement('div');
    bloque.className = 'entrada-historial';
    bloque.innerHTML = `
      <div class="meta-linea">${data.fecha} · Archivo: ${data.archivo}</div>
      <div class="formula-orig">Fórmula original: ${escaparHtml(data.formula_original)}</div>
      <div class="pasos-texto">${escaparHtml(data.resolucion)}</div>
    `;
    cuadricula.prepend(bloque);
    cuadricula.scrollTop = 0;
    estado2.textContent = 'Resuelto y añadido a la cuadrícula inferior.';
  } catch (e) {
    estado2.textContent = 'Error al intentar solucionar.';
  }
}

function escaparHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

document.addEventListener('DOMContentLoaded', renderizarTeclado);
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, historial=historial_sesiones)


@app.route("/api/descripcion_sintactica", methods=["POST"])
def descripcion_sintactica():
    payload = request.get_json(silent=True) or {}
    formula = (payload.get("formula") or "").strip()
    if not formula:
        return jsonify({"descripcion": ""})

    prompt = (
        "Actúa estrictamente como un analizador sintáctico matemático (parser).\n"
        "Tu ÚNICA misión es leer la siguiente expresión y desglosar paso a paso su estructura sintáctica, "
        "la jerarquía de operadores, el alcance de los paréntesis, raíces, fracciones y funciones.\n"
        "REGLAS ESTRICTAS:\n"
        "- PROHIBIDO calcular, simplificar o dar ningún resultado numérico.\n"
        "- Describe de izquierda a derecha qué operación engloba a cada término para evitar ambigüedades.\n"
        "- Responde en español claro, conciso y numerado paso a paso.\n\n"
        f"Expresión a analizar:\n{formula}"
    )
    fallback = f"Lectura literal de tokens de izquierda a derecha: {formula}"
    desc = limpiar_salida_texto(llamar_motor_ia(prompt, fallback))
    return jsonify({"descripcion": desc})


@app.route("/api/a_latex", methods=["POST"])
def convertir_a_latex():
    payload = request.get_json(silent=True) or {}
    formula = (payload.get("formula") or "").strip()
    descripcion = (payload.get("descripcion") or "").strip()
    if not formula:
        return jsonify({"latex": ""})

    bloque_desc = ""
    if descripcion:
        bloque_desc = (
            f"\nUsa como guía obligatoria este desglose sintáctico ya validado:\n{descripcion}\n"
        )

    prompt = (
        "Convierte la siguiente expresión o ecuación matemática con símbolos científicos "
        "a una expresión matemática limpia y legible en LaTeX.\n"
        "No calcules ni resuelvas nada. Devuelve únicamente el LaTeX resultante, sin explicaciones extra ni bloques de código:\n"
        f"Expresión original: {formula}"
        f"{bloque_desc}"
    )
    fallback = formula
    latex_limpio = llamar_motor_ia(prompt, fallback)
    return jsonify({"latex": latex_limpio})


@app.route("/api/solucionar", methods=["POST"])
def solucionar():
    payload = request.get_json(silent=True) or {}
    latex_expr = (payload.get("latex") or "").strip()
    formula_orig = (payload.get("formula_original") or latex_expr).strip()
    desc_sintactica = (payload.get("descripcion_sintactica") or "").strip()

    valor_sympy = calcular_exacto_sympy(formula_orig, latex_expr)
    bloque_verificacion = ""
    if valor_sympy:
        bloque_verificacion = (
            f"\nRESULTADO EXACTO VERIFICADO SIMBÓLICAMENTE: {valor_sympy}\n"
            "Debes respetar este resultado exacto al 100 % y asegurarte de que todos los pasos conduzcan exactamente a él.\n"
        )

    prompt = (
        "Resuelve la siguiente expresión o ecuación matemática y explica cada paso.\n"
        f"{bloque_verificacion}"
        "REGLAS ESTRICTAS DE FORMATO:\n"
        "- No uses etiquetas LaTeX ni signos de dólar ($ o $$).\n"
        "- No uses almohadillas (#) ni asteriscos (**) ni barras invertidas (\\).\n"
        "- Usa símbolos matemáticos limpios y legibles: √, π, ×, ÷, ², ³, ·\n"
        "- Convención de calculadora científica: log() es logaritmo en base 10 y ln() es logaritmo neperiano.\n"
        "- No dudes ni te corrija en el texto: entrega la solución definitiva y coherente paso a paso.\n\n"
        "Estructura exacta de respuesta:\n"
        "Resultado final: [valor o expresión final exacta]\n\n"
        "Desglose paso a paso:\n"
        "Paso 1: ...\n"
        "Paso 2: ...\n\n"
        f"Expresión original: {formula_orig}\n"
        f"Expresión LaTeX: {latex_expr}"
    )
    fallback = (
        f"Resultado final: {valor_sympy if valor_sympy else 'Sin conexión con FreeLLMAPI'}\n\n"
        f"Expresión recibida: {latex_expr}\n"
        "Verifica que FreeLLMAPI esté activo en http://127.0.0.1:3001."
    )
    resolucion_cruda = llamar_motor_ia(prompt, fallback)
    resolucion = limpiar_salida_texto(resolucion_cruda)
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registro = {
        "fecha": fecha,
        "formula_original": formula_orig,
        "descripcion_sintactica": desc_sintactica,
        "latex_agente_1": latex_expr,
        "resolucion": resolucion,
    }
    archivo = guardar_auditoria(registro)
    registro["archivo"] = archivo
    historial_sesiones.insert(0, registro)
    return jsonify(registro)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
