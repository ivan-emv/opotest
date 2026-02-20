import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# =========================
# Config
# =========================
st.set_page_config(page_title="Cuestionarios Oposición", page_icon="📚", layout="wide")
DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/1ThQ4esYrhRrQuqv6NSIPOFd4aGxRdGMG/view?usp=drive_link"

# =========================
# Modelos
# =========================
@dataclass
class Question:
    qnum: int
    text: str
    options: List[str]  # ["a) ...", "b) ...", ...]
    page: int           # 1-based

@dataclass
class Section:
    title: str
    start_page: int
    end_page: int
    questions: Dict[int, Question]      # qnum -> Question
    answers: Dict[int, str]             # qnum -> "a"/"b"/"c"/"d"
    solutions_page: int                 # 1-based


# =========================
# Google Drive (descarga pública)
# =========================
def extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def download_drive_file(file_id: str, timeout: int = 60) -> bytes:
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    r = session.get(url, params={"id": file_id}, stream=True, timeout=timeout)
    r.raise_for_status()

    if "content-disposition" in r.headers:
        return r.content

    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if not token:
        m = re.search(r"confirm=([0-9A-Za-z_]+)", r.text)
        if m:
            token = m.group(1)

    if not token:
        raise RuntimeError(
            "Google Drive no permite descarga directa. Verifica que el archivo esté compartido como "
            "'Cualquiera con el enlace' y sea descargable."
        )

    r2 = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=timeout)
    r2.raise_for_status()
    return r2.content


# =========================
# PDF Utils
# =========================
@st.cache_data(show_spinner=False)
def pdf_num_pages(pdf_bytes: bytes) -> int:
    if fitz is None:
        raise RuntimeError("Falta PyMuPDF (pymupdf). Revisa requirements.txt.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = doc.page_count
    doc.close()
    return n


@st.cache_data(show_spinner=False)
def extract_page_texts(pdf_bytes: bytes) -> List[str]:
    """Devuelve lista de textos por página (0-based index)."""
    if fitz is None:
        raise RuntimeError("Falta PyMuPDF (pymupdf). Revisa requirements.txt.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(doc.page_count):
        t = doc.load_page(i).get_text("text") or ""
        # normalización básica
        t = t.replace("\r", "\n")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        pages.append(t.strip())
    doc.close()
    return pages


# =========================
# Parser: Preguntas + Soluciones
# =========================
Q_START_RE = re.compile(r"(?m)^\s*(\d+)\.\s+(.*)$")
OPT_RE = re.compile(r"(?m)^\s*([a-dA-D])\)\s*(.+)$")
SOLUTIONS_HEADER_RE = re.compile(r"(?im)^\s*Soluciones\s*$")
SOLUTION_PAIR_RE = re.compile(r"(?i)\b(\d+)\.\s*([a-d])\b")

def extract_title_from_page(text: str) -> str:
    """
    Heurística: el título suele estar al inicio, antes de que empiecen los números de pregunta.
    """
    if not text:
        return "Sin título"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "Sin título"

    # recorta a primeras 6 líneas hasta que aparezca "1."
    out = []
    for l in lines[:10]:
        if re.match(r"^\d+\.\s", l):
            break
        out.append(l)
        if len(out) >= 6:
            break
    title = " ".join(out).strip()
    return title if title else "Sin título"


def parse_questions_from_text(text: str, page_num_1based: int) -> Dict[int, Question]:
    """
    Extrae preguntas en formato:
      1. Enunciado
         a) ...
         b) ...
         c) ...
         d) ...
    Devuelve dict qnum -> Question
    """
    if not text:
        return {}

    # Si es página de soluciones, no extraer preguntas
    if SOLUTIONS_HEADER_RE.search(text):
        return {}

    lines = text.split("\n")
    questions: Dict[int, Question] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not m:
            i += 1
            continue

        qnum = int(m.group(1))
        qtext = m.group(2).strip()

        # Captura líneas siguientes hasta encontrar opciones o próxima pregunta
        j = i + 1
        buffer = []
        opts = []

        # Primero, acumular texto adicional del enunciado hasta que empiecen opciones (a)/b)/...)
        while j < len(lines):
            l = lines[j].strip()
            if re.match(r"^\d+\.\s+", l):
                break
            if re.match(r"^[a-dA-D]\)\s+", l):
                break
            if l:
                buffer.append(l)
            j += 1

        if buffer:
            qtext = (qtext + " " + " ".join(buffer)).strip()

        # Ahora, capturar opciones a-d
        k = j
        seen = set()
        while k < len(lines):
            l = lines[k].strip()
            if re.match(r"^\d+\.\s+", l):
                break
            om = re.match(r"^([a-dA-D])\)\s*(.+)$", l)
            if om:
                letter = om.group(1).lower()
                opt_text = om.group(2).strip()
                if letter not in seen:
                    opts.append(f"{letter}) {opt_text}")
                    seen.add(letter)
            k += 1

        # Validación: necesitamos 4 opciones
        if len(opts) == 4:
            questions[qnum] = Question(
                qnum=qnum,
                text=qtext,
                options=opts,
                page=page_num_1based,
            )

        i = k

    return questions


def parse_solutions_from_text(text: str) -> Dict[int, str]:
    """
    Extrae soluciones como:
      Soluciones
      1. a  4. d  7. c ...
    Devuelve dict qnum -> letra
    """
    if not text:
        return {}
    if not SOLUTIONS_HEADER_RE.search(text):
        return {}

    pairs = SOLUTION_PAIR_RE.findall(text)
    ans: Dict[int, str] = {}
    for qn, letter in pairs:
        ans[int(qn)] = letter.lower()
    return ans


@st.cache_data(show_spinner=False)
def build_sections(pdf_bytes: bytes) -> List[Section]:
    """
    Construye secciones detectando el patrón:
      [páginas con preguntas] -> [página con "Soluciones"].
    """
    pages = extract_page_texts(pdf_bytes)
    n = len(pages)

    all_sections: List[Section] = []

    current_questions: Dict[int, Question] = {}
    current_title: Optional[str] = None
    start_page = 1

    for idx in range(n):
        page_num = idx + 1
        text = pages[idx]

        # soluciones
        sol = parse_solutions_from_text(text)
        if sol:
            # Cerrar sección si tenemos preguntas acumuladas
            if current_questions:
                title = current_title or "Cuestionario"
                # Filtrar solo preguntas que tengan respuesta
                q_with_ans = {qn: q for qn, q in current_questions.items() if qn in sol}
                ans_filtered = {qn: sol[qn] for qn in q_with_ans.keys()}

                if q_with_ans and ans_filtered:
                    all_sections.append(
                        Section(
                            title=title,
                            start_page=start_page,
                            end_page=page_num,
                            questions=q_with_ans,
                            answers=ans_filtered,
                            solutions_page=page_num,
                        )
                    )

            # Reset acumuladores: la siguiente sección empieza tras soluciones
            current_questions = {}
            current_title = None
            start_page = page_num + 1
            continue

        # Si no es soluciones, extraer preguntas
        qs = parse_questions_from_text(text, page_num)
        if qs:
            # título se toma del primer bloque que tenga preguntas
            if current_title is None:
                current_title = extract_title_from_page(text)
            # merge
            current_questions.update(qs)

    return all_sections


# =========================
# Quiz engine
# =========================
def letter_to_index(letter: str) -> int:
    return {"a": 0, "b": 1, "c": 2, "d": 3}.get(letter.lower(), -1)


def index_to_letter(idx: int) -> str:
    return ["a", "b", "c", "d"][idx] if 0 <= idx <= 3 else "?"


# =========================
# Estado
# =========================
def init_state():
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "source_label" not in st.session_state:
        st.session_state.source_label = None

    if "sections" not in st.session_state:
        st.session_state.sections = []
    if "quiz" not in st.session_state:
        st.session_state.quiz = []  # List[Tuple[Section, Question]]
    if "answers" not in st.session_state:
        st.session_state.answers = {}  # key -> selected idx
    if "seed" not in st.session_state:
        st.session_state.seed = random.randint(1, 10_000_000)

init_state()

# =========================
# UI
# =========================
st.title("📚 Cuestionarios Oposición (desde tu PDF, con soluciones)")
st.caption("Operativa: carga PDF → detecta secciones Preguntas/Soluciones → genera test → corrige y reporta métricas.")

with st.sidebar:
    st.header("1) Carga del PDF")

    drive_url = st.text_input("Enlace Google Drive (público)", value=DEFAULT_DRIVE_URL)
    col1, col2 = st.columns(2)
    with col1:
        btn_drive = st.button("📥 Cargar Drive", use_container_width=True)
    with col2:
        btn_clear = st.button("🧹 Limpiar", use_container_width=True)

    st.divider()
    up = st.file_uploader("Subir PDF manualmente", type=["pdf"], accept_multiple_files=False)

    st.divider()
    st.header("2) Configuración del test")
    num_q = st.slider("Número de preguntas", 5, 80, 20, 1)
    st.session_state.seed = st.number_input("Semilla (barajado)", value=int(st.session_state.seed), step=1)

if btn_clear:
    st.session_state.pdf_bytes = None
    st.session_state.source_label = None
    st.session_state.sections = []
    st.session_state.quiz = []
    st.session_state.answers = {}
    st.session_state.seed = random.randint(1, 10_000_000)
    st.rerun()

# Prioridad: subida manual
if up is not None:
    st.session_state.pdf_bytes = up.read()
    st.session_state.source_label = f"Archivo subido: {up.name}"

if btn_drive:
    fid = extract_drive_file_id(drive_url)
    if not fid:
        st.error("No se pudo extraer el ID del archivo del enlace de Drive.")
    else:
        with st.spinner("Descargando PDF desde Google Drive..."):
            try:
                b = download_drive_file(fid)
                st.session_state.pdf_bytes = b
                st.session_state.source_label = f"Google Drive file_id: {fid}"
                st.success("PDF cargado correctamente.")
            except Exception as e:
                st.error(f"No se pudo descargar desde Drive: {e}")

if st.session_state.pdf_bytes is None:
    st.info("Cargue el PDF para iniciar el procesamiento.")
    st.stop()

# Procesado: construir secciones
with st.spinner("Analizando PDF: detección de preguntas y soluciones..."):
    try:
        st.session_state.sections = build_sections(st.session_state.pdf_bytes)
    except Exception as e:
        st.error(f"Error procesando el PDF: {e}")
        st.stop()

if not st.session_state.sections:
    st.error(
        "No se han detectado secciones Preguntas/Soluciones. "
        "Esto puede ocurrir si el PDF es escaneado (sin texto) o si el formato no coincide."
    )
    st.stop()

# =========================
# Panel principal
# =========================
left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("📄 Fuente")
    st.write(st.session_state.source_label or "PDF cargado")

    n_pages = pdf_num_pages(st.session_state.pdf_bytes)
    st.caption(f"Páginas: {n_pages} | Secciones detectadas: {len(st.session_state.sections)}")

    st.markdown("### Secciones detectadas")
    section_labels = []
    for i, sec in enumerate(st.session_state.sections, start=1):
        qcount = len(sec.questions)
        section_labels.append(
            f"{i:02d}. {sec.title}  |  Preguntas: {qcount}  |  Páginas: {sec.start_page}-{sec.solutions_page}"
        )

    selected = st.multiselect(
        "Seleccione una o varias secciones",
        options=section_labels,
        default=[section_labels[0]] if section_labels else [],
    )

    gen = st.button("⚙️ Generar test", type="primary", use_container_width=True)

    if gen:
        # Map selección a secciones
        selected_idxs = []
        label_to_idx = {label: i for i, label in enumerate(section_labels)}
        for lab in selected:
            if lab in label_to_idx:
                selected_idxs.append(label_to_idx[lab])

        pool: List[Tuple[Section, Question]] = []
        for idx in selected_idxs:
            sec = st.session_state.sections[idx]
            for qn, q in sec.questions.items():
                if qn in sec.answers:
                    pool.append((sec, q))

        if not pool:
            st.error("No hay preguntas disponibles en la selección actual.")
        else:
            rnd = random.Random(int(st.session_state.seed))
            rnd.shuffle(pool)
            st.session_state.quiz = pool[: min(len(pool), int(num_q))]
            st.session_state.answers = {}
            st.success(f"Test generado: {len(st.session_state.quiz)} preguntas.")

    if st.session_state.quiz:
        st.divider()
        colx, coly = st.c
