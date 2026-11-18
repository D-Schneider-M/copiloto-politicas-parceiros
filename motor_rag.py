"""Motor de RAG enxuto: embeddings do Gemini + similaridade com numpy + saída estruturada (Pydantic)."""

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()  # lê GEMINI_API_KEY do arquivo .env, se existir

# Modelos configuráveis por variável de ambiente (caso algum seja descontinuado)
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")


# ---------- Saída estruturada ----------
class TriagemChamado(BaseModel):
    categoria: Literal[
        "repasse", "cancelamento", "atraso_entrega", "reembolso", "item_indisponivel", "outros"
    ] = Field(description="Categoria principal do chamado.")
    urgencia: Literal["baixa", "media", "alta"] = Field(
        description="Urgência. Casos de segurança alimentar são sempre 'alta'."
    )
    elegivel_reembolso: Optional[bool] = Field(
        description="True se o restaurante tem direito a receber/ser ressarcido pelo valor do pedido; "
        "False se não tem; null se a política não permite concluir."
    )
    clausula_citada: str = Field(
        description="Cláusula principal que fundamenta a decisão, ex: 'Cláusula 2.3'. Use 'Nenhuma' se não houver."
    )
    precisa_atendimento_humano: bool = Field(
        description="True se o caso não é coberto pela política ou exige análise humana."
    )
    resposta_ao_parceiro: str = Field(
        description="Resposta cordial em português para o restaurante, em até 5 frases, citando a cláusula."
    )


# ---------- Base de conhecimento ----------
@dataclass
class Trecho:
    id: str
    secao: str
    texto: str


def carregar_trechos(texto: str) -> list[Trecho]:
    """Divide a política em cláusulas. Se não houver 'Cláusula', divide por parágrafos."""
    trechos: list[Trecho] = []
    secao, atual = "", None
    for linha in texto.splitlines():
        l = linha.strip()
        if not l:
            continue
        if re.match(r"^\d+\.\s+\S", l):  # título de seção, ex: "2. CANCELAMENTOS"
            secao = l
        elif l.startswith("Cláusula"):
            if atual:
                trechos.append(atual)
            m = re.match(r"Cláusula\s+(\d+(?:\.\d+)*)", l)
            atual = Trecho(id=m.group(1) if m else str(len(trechos) + 1), secao=secao, texto=l)
        elif atual:
            atual.texto += " " + l
    if atual:
        trechos.append(atual)

    if not trechos:  # fallback para PDFs/textos sem estrutura de cláusulas
        paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if len(p.strip()) > 40]
        trechos = [Trecho(id=f"P{i + 1}", secao="", texto=p) for i, p in enumerate(paragrafos)]
    return trechos


def _embed(client: genai.Client, textos: list[str], task_type: str) -> np.ndarray:
    vetores = []
    for i in range(0, len(textos), 90):  # lotes pequenos para respeitar limites da API
        r = client.models.embed_content(
            model=EMBED_MODEL,
            contents=textos[i : i + 90],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vetores.extend(e.values for e in r.embeddings)
    m = np.array(vetores, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)  # normaliza: produto escalar = cosseno


class BaseConhecimento:
    def __init__(self, texto: str, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Defina GEMINI_API_KEY (variável de ambiente) ou informe a chave no app.")
        self.client = genai.Client(api_key=key)
        self.trechos = carregar_trechos(texto)
        self.matriz = _embed(
            self.client, [f"{t.secao}\n{t.texto}" for t in self.trechos], "RETRIEVAL_DOCUMENT"
        )

    def buscar(self, pergunta: str, k: int = 3) -> list[tuple[Trecho, float]]:
        q = _embed(self.client, [pergunta], "RETRIEVAL_QUERY")[0]
        scores = self.matriz @ q
        melhores = np.argsort(scores)[::-1][:k]
        return [(self.trechos[i], float(scores[i])) for i in melhores]

    def triar(self, chamado: str, k: int = 3) -> tuple[TriagemChamado, list[tuple[Trecho, float]]]:
        encontrados = self.buscar(chamado, k)
        contexto = "\n".join(f"[Cláusula {t.id}] {t.texto}" for t, _ in encontrados)

        instrucoes = (
            "Você é um assistente de suporte a restaurantes parceiros. Decida SOMENTE com base nas "
            "cláusulas fornecidas. Não invente regras. Se as cláusulas não cobrem o caso, use "
            "categoria 'outros', clausula_citada 'Nenhuma' e precisa_atendimento_humano true. "
            "Se o chamado não informar dados necessários (ex: quem causou o atraso, plano do restaurante), "
            "diga isso na resposta em vez de assumir."
        )
        prompt = f"CLÁUSULAS RECUPERADAS:\n{contexto}\n\nCHAMADO DO RESTAURANTE:\n{chamado}"

        resp = self.client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instrucoes,
                response_mime_type="application/json",
                response_schema=TriagemChamado,
                temperature=0.1,
            ),
        )
        triagem = resp.parsed or TriagemChamado.model_validate_json(resp.text)
        return triagem, encontrados


# ---------- Teste rápido no terminal ----------
if __name__ == "__main__":
    with open("politica_parceiros.txt", encoding="utf-8") as f:
        base = BaseConhecimento(f.read())
    chamado = "O entregador atrasou 40 min e o cliente cancelou. Vou receber pelo pedido? Uso o plano Entrega Parceira."
    triagem, trechos = base.triar(chamado)
    print("Trechos:", [(t.id, round(s, 3)) for t, s in trechos])
    print(triagem.model_dump_json(indent=2))
