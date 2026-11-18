import hashlib
import os
from pathlib import Path

import streamlit as st

from motor_rag import BaseConhecimento

st.set_page_config(page_title="Copiloto de Políticas para Parceiros", page_icon="🍽️", layout="wide")

CHAMADOS_EXEMPLO = {
    "Atraso do entregador + cancelamento": "O entregador atrasou 40 minutos e o cliente cancelou o pedido. Marcamos como pronto dentro do tempo informado. Uso o plano Entrega Parceira. Vou receber pelo pedido?",
    "Item faltante": "O cliente disse que faltou uma bebida no pedido e mandou foto no mesmo dia. Vou ter que reembolsar?",
    "Quando cai meu repasse": "Em que dia da semana recebo o repasse dos pedidos concluídos? Posso antecipar?",
    "Suspeita de contaminação": "Um cliente alegou que encontrou um corpo estranho na comida e pediu reembolso integral. O que acontece agora?",
    "Cancelei por falta de item": "Cancelei um pedido porque acabou o ingrediente principal e não deu tempo de pausar o item. Serei penalizado?",
    "Fora da política": "Posso alterar o horário de funcionamento do meu restaurante nos feriados pelo app?",
}


def get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


@st.cache_resource(show_spinner="Gerando embeddings da política...")
def carregar_base(texto: str, api_key: str, _hash: str) -> BaseConhecimento:
    return BaseConhecimento(texto, api_key)


def ler_upload(arquivo) -> str:
    if arquivo.name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n\n".join(p.extract_text() or "" for p in PdfReader(arquivo).pages)
    return arquivo.getvalue().decode("utf-8")


st.title("🍽️ Copiloto de Políticas e Triagem de Chamados")
st.caption(
    "Projeto de portfólio: RAG + saída estruturada (JSON) com Gemini. "
    "A política usada é **fictícia** e serve apenas para demonstração."
)

with st.sidebar:
    st.header("Configuração")
    api_key = get_api_key() or st.text_input("Gemini API Key", type="password")
    upload = st.file_uploader("Usar outra política (.txt ou .pdf)", type=["txt", "pdf"])
    k = st.slider("Trechos recuperados (top-k)", 1, 5, 3)

if upload:
    texto_politica = ler_upload(upload)
else:
    texto_politica = Path("politica_parceiros.txt").read_text(encoding="utf-8")

with st.expander("Ver política em uso"):
    st.text(texto_politica)

if not api_key:
    st.info("Informe a Gemini API Key na barra lateral para começar.")
    st.stop()

escolha = st.selectbox("Chamado de exemplo", ["(digitar o meu)"] + list(CHAMADOS_EXEMPLO))
padrao = CHAMADOS_EXEMPLO.get(escolha, "")
chamado = st.text_area("Chamado do restaurante parceiro", value=padrao, height=120)

if st.button("Analisar chamado", type="primary") and chamado.strip():
    try:
        base = carregar_base(texto_politica, api_key, hashlib.md5(texto_politica.encode()).hexdigest())
        with st.spinner("Recuperando cláusulas e gerando a triagem..."):
            triagem, trechos = base.triar(chamado, k)
    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("JSON para o sistema interno")
        st.json(triagem.model_dump())
    with col2:
        st.subheader("Resposta pronta para o restaurante")
        st.success(triagem.resposta_ao_parceiro)
        if triagem.precisa_atendimento_humano:
            st.warning("Encaminhar para atendimento humano.")

    st.subheader("Trechos recuperados (RAG)")
    for t, score in trechos:
        with st.expander(f"Cláusula {t.id} — similaridade {score:.3f}"):
            st.write(t.texto)
