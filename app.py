import re
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import streamlit as st

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# =========================
# Configuración
# =========================
st.set_page_config(page_title="Cuestionarios Oposición", page_icon="📚", layout="wide")

DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/1ThQ4esYrhRrQuqv6NSIPOFd4aGxRdGMG/view?usp=drive_link"


# =========================
# Modelos
# =========================
@dataclass
class MCQ:
    question: str
    options: List[str]
    correct_index: int
    source_sentence: str


# =========================
# Google Drive: descarga pública
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

    # descarga directa
    if "content-disposition" in r.headers:
        return r.content

    # token confirmación
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
            "No se pudo obtener el token de descarga de Google Drive. "
            "Verifica que el archivo esté compartido como 'Cualquiera con el enlace'."
        )

    r2 = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=timeout)
    r2.raise_for_status()
    return r2.content


# =========================
# PDF -> Texto
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
def extract_text_from_pdf(pdf_bytes: bytes, start_page: int, end_page: int) -> str:
    if fitz is None:
        raise RuntimeError("Falta PyMuPDF (pymupdf). Revisa requirements.txt.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = doc.page_count

    sp = max(1, min(start_page, n))
    ep = max(1, min(end_page, n))
    if ep < sp:
        sp, ep = ep, sp

    chunks = []
    for i in range(sp - 1, ep):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        chunks.append(txt)

    doc.close()
    text = "\n".join(chunks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    # Segmentación sencilla (sin dependencias NLP)
    # Mejorable, pero estable para un MVP.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!;:])\s+", text)
    # Filtrar oraciones demasiado cortas o demasiado largas
    out = []
    for s in parts:
        s = s.strip()
        if 60 <= len(s) <= 220:
            out.append(s)
    return out


def extract_candidate_terms(text: str) -> List[str]:
    """
    Extrae términos candidatos para usar como respuestas:
    - secuencias de palabras con mayúsculas iniciales (p.ej. "Administración Pública")
    - siglas (p.ej. "BOE", "UE")
    - palabras "técnicas" largas (>= 8)
    - números con formato típico (p.ej. "15", "10%", "2024")
    """
    terms = set()

    # Siglas
    for m in re.findall(r"\b[A-ZÁÉÍÓÚÜÑ]{2,}\b", text):
        if 2 <= len(m) <= 10:
            terms.add(m)

    # Capitalizadas multi-palabra (hasta 4 palabras)
    for m in re.findall(r"\b(?:[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+){1,3}\b", text):
        if 6 <= len(m) <= 45:
            terms.add(m.strip())

    # Palabras largas (técnicas)
    for m in re.findall(r"\b[a-záéíóúüñ]{8,}\b", text.lower()):
        # evitar ruido muy común
        if m not in {"mediante", "respecto", "establecer", "procedimiento", "administración"}:
            terms.add(m)

    # Números / porcentajes / años
    for m in re.findall(r"\b\d{1,4}(?:[.,]\d{1,2})?(?:%|)\b", text):
        terms.add(m)

    # limpieza
    cleaned = []
    for t in terms:
        t = t.strip(" ,.;:()[]{}\"'“”‘’")
        if 2 <= len(t) <= 50:
            cleaned.append(t)

    # orden estable por longitud (ayuda a escoger términos “más sustantivos”)
    cleaned.sort(key=lambda x: (len(x), x))
    return cleaned


def choose_blank_term(sentence: str, global_terms: List[str]) -> Optional[str]:
    """
    Escoge un término a ocultar dentro de una oración.
    Priorizamos coincidencias de términos globales presentes en la oración.
    """
    s = sentence

    present = []
    s_lower = s.lower()
    for t in global_terms:
        # comparación robusta
        if t.lower() in s_lower and len(t) >= 3:
            present.append(t)

    # priorizar términos más largos y “con aspecto” de concepto
    present.sort(key=lambda x: len(x), reverse=True)

    # evitar ocultar palabras extremadamente comunes
    stop = {"artículo", "articulos", "capítulo", "capitulo", "sección", "seccion", "disposición", "disposicion"}
    for t in present:
        if t.lower() in stop:
            continue
        # exigir que no sea parte minúscula suelta
        if len(t) >= 4:
            return t

    return None


def make_question_from_sentence(sentence: str, answer: str) -> str:
    # reemplazo de la primera ocurrencia (case-insensitive aproximado)
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    blanked = pattern.sub("_____ ", sentence, count=1)
    return f"Completa la frase:\n\n{blanked}"


def generate_mcqs(text: str, num_questions: int, seed: int = 1234) -> List[MCQ]:
    rng = random.Random(seed)

    sentences = split_sentences(text)
    if len(sentences) < 5:
        return []

    terms = extract_candidate_terms(text)
    if len(terms) < 10:
        return []

    # muestreo de oraciones aleatorio para variedad
    rng.shuffle(sentences)

    mcqs: List[MCQ] = []
    used_pairs = set()

    for s in sentences:
        if len(mcqs) >= num_questions:
            break

        ans = choose_blank_term(s, terms)
        if not ans:
            continue

        key = (s[:80], ans.lower())
        if key in used_pairs:
            continue
        used_pairs.add(key)

        # distractores: otros términos similares
        # filtramos para que no aparezcan dentro de la oración (evita pistas)
        distract_pool = [t for t in terms if t.lower() != ans.lower() and t.lower() not in s.lower()]
        if len(distract_pool) < 3:
            continue

        # preferir distractores de longitud parecida (mejor calidad)
        distract_pool.sort(key=lambda x: abs(len(x) - len(ans)))
        distractors = []
        for t in distract_pool:
            if t.lower() == ans.lower():
                continue
            if t in distractors:
                continue
            distractors.append(t)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue

        options = [ans] + distractors
        rng.shuffle(options)
        correct_index = options.index(ans)

        q = MCQ(
            question=make_question_from_sentence(s, ans),
            options=options,
            correct_index=correct_index,
            source_sentence=s,
        )
        mcqs.append(q)

    return mcqs


# =========================
# Estado
# =========================
def init_state():
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "source_label" not in st.session_state:
        st.session_state.source_label = None
    if "mcqs" not in st.session_state:
        st.session_state.mcqs = []
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "seed" not in st.session_state:
        st.session_state.seed = random.randint(1, 10_000_000)


init_state()


# =========================
# UI
# =========================
st.title("📚 Cuestionarios para Oposición (Automático, sin IA)")
st.caption("Generación heurística tipo test: completa la frase, corrección y estadísticas.")

with st.sidebar:
    st.header("1) Temario (PDF)")
    drive_url = st.text_input("Enlace Google Drive (público)", value=DEFAULT_DRIVE_URL)
    col_a, col_b = st.columns(2)
    with col_a:
        load_drive = st.button("📥 Cargar Drive", use_container_width=True)
    with col_b:
        clear_all = st.button("🧹 Limpiar", use_container_width=True)

    st.divider()
    up = st.file_uploader("Subir PDF manualmente", type=["pdf"], accept_multiple_files=False)

    st.divider()
    st.header("2) Configuración")
    num_q = st.slider("Número de preguntas", 5, 50, 15, 1)
    st.session_state.seed = st.number_input("Semilla (barajar/generación)", value=int(st.session_state.seed), step=1)

    st.divider()
    st.caption("Consejo operativo: si el PDF es escaneado (imagen), no habrá texto extraíble.")


if clear_all:
    st.session_state.pdf_bytes = None
    st.session_state.source_label = None
    st.session_state.mcqs = []
    st.session_state.answers = {}
    st.session_state.seed = random.randint(1, 10_000_000)
    st.rerun()

if up is not None:
    st.session_state.pdf_bytes = up.read()
    st.session_state.source_label = f"Archivo subido: {up.name}"

if load_drive:
    fid = extract_drive_file_id(drive_url)
    if not fid:
        st.error("No se pudo extraer el ID del archivo desde el enlace.")
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
    st.info("Cargue el PDF (Drive o subida manual) para comenzar.")
    st.stop()

left, right = st.columns([0.44, 0.56], gap="large")

with left:
    st.subheader("📄 Documento cargado")
    st.write(st.session_state.source_label or "Fuente: PDF")

    try:
        n_pages = pdf_num_pages(st.session_state.pdf_bytes)
        st.caption(f"Páginas detectadas: {n_pages}")
    except Exception as e:
        st.error(f"No se pudo leer el PDF: {e}")
        st.stop()

    st.markdown("### Selección de páginas")
    p1, p2 = st.slider("Rango de páginas", 1, n_pages, (1, min(n_pages, 10)), 1)

    with st.expander("Vista previa del texto extraído", expanded=False):
        try:
            preview = extract_text_from_pdf(st.session_state.pdf_bytes, p1, p2)
            st.text_area("Texto (solo lectura)", value=preview[:6000], height=240, disabled=True)
            if not preview.strip():
                st.warning("No se extrajo texto. Si el PDF es escaneado, este MVP no podrá generar preguntas.")
        except Exception as e:
            st.error(f"Error extrayendo texto: {e}")

    st.markdown("### Generación del cuestionario")
    gen = st.button("⚙️ Generar preguntas", type="primary", use_container_width=True)

    if gen:
        with st.spinner("Extrayendo texto y generando preguntas (heurístico)..."):
            try:
                text = extract_text_from_pdf(st.session_state.pdf_bytes, p1, p2)
                if not text.strip():
                    st.error("No hay texto extraíble en el rango seleccionado.")
                else:
                    mcqs = generate_mcqs(text=text, num_questions=num_q, seed=int(st.session_state.seed))
                    if not mcqs:
                        st.error(
                            "No se pudieron generar preguntas con el texto seleccionado. "
                            "Pruebe ampliando el rango de páginas o eligiendo un apartado más denso en conceptos."
                        )
                    else:
                        st.session_state.mcqs = mcqs
                        st.session_state.answers = {}
                        st.success(f"Cuestionario generado: {len(mcqs)} preguntas.")
            except Exception as e:
                st.error(str(e))

    if st.session_state.mcqs:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Barajar preguntas", use_container_width=True):
                rnd = random.Random(int(st.session_state.seed) + 999)
                rnd.shuffle(st.session_state.mcqs)
                st.session_state.answers = {}
                st.rerun()
        with col2:
            if st.button("🧾 Reiniciar respuestas", use_container_width=True):
                st.session_state.answers = {}
                st.rerun()


with right:
    st.subheader("📝 Test")
    if not st.session_state.mcqs:
        st.info("Genere un cuestionario para comenzar.")
        st.stop()

    for i, q in enumerate(st.session_state.mcqs, start=1):
        st.markdown(f"**{i}. {q.question}**")

        key = f"q_{i}"
        prev = st.session_state.answers.get(key, None)
        idx = 0 if prev is None else (prev + 1)

        choice = st.radio(
            label="",
            options=["— Seleccione una opción —"] + q.options,
            index=idx,
            key=f"radio_{i}",
            label_visibility="collapsed",
        )

        if choice == "— Seleccione una opción —":
            st.session_state.answers[key] = None
        else:
            st.session_state.answers[key] = q.options.index(choice)

        st.divider()

    st.markdown("### ✅ Corrección y resultados")
    corregir = st.button("Calcular resultado", type="primary", use_container_width=True)

    if corregir:
        total = len(st.session_state.mcqs)
        answered = 0
        correct = 0
        incorrect = 0
        blank = 0

        for i, q in enumerate(st.session_state.mcqs, start=1):
            ans = st.session_state.answers.get(f"q_{i}")
            if ans is None:
                blank += 1
                continue
            answered += 1
            if ans == q.correct_index:
                correct += 1
            else:
                incorrect += 1

        nota = round((correct / total) * 10, 2) if total else 0.0

        st.info(
            f"Total: {total} | Respondidas: {answered} | "
            f"Aciertos: {correct} | Errores: {incorrect} | Sin responder: {blank} | "
            f"Nota: {nota}/10"
        )

        with st.expander("Revisión detallada (respuestas correctas)", expanded=True):
            for i, q in enumerate(st.session_state.mcqs, start=1):
                ans = st.session_state.answers.get(f"q_{i}")
                correct_opt = q.options[q.correct_index]
                user_opt = "— Sin responder —" if ans is None else q.options[ans]

                if ans == q.correct_index:
                    st.success(f"{i}) Correcta ✅ | Tu respuesta: {user_opt}")
                else:
                    st.error(f"{i}) Incorrecta ❌ | Tu respuesta: {user_opt} | Correcta: {correct_opt}")

                # Fuente para estudiar (trazabilidad)
                st.caption(f"Frase original: {q.source_sentence}")
                st.markdown("---")

