# 🍽️ Copiloto de Políticas e Triagem de Chamados com IA

Sistema de **RAG (Retrieval-Augmented Generation)** com **Saídas Estruturadas (JSON)** construído com a API do **Google Gemini** e **Streamlit** para automatizar a triagem de chamados de restaurantes parceiros em operações de delivery.

**App online:** [Acessar o Copiloto de Políticas](https://copiloto-politicas-parceiros-keeeyqdvvkpbncncfwaa9t.streamlit.app)

> A base padrão (`politica_parceiros.txt`) contém regras **fictícias** de repasse, cancelamento, atrasos e reembolso criadas para demonstração. Também é possível enviar qualquer outro regulamento em `.txt` ou `.pdf` diretamente pela interface.

---

## 🎯 Problema de Negócio Resolvido

Operações de marketplace e delivery recebem diariamente milhares de solicitações de parceiros envolvendo disputas de cancelamento, atrasos logísticos, repasses financeiros e segurança alimentar. Respostas manuais são lentas e inconsistentes, enquanto LLMs sem restrições correm o risco de inventar regras comerciais.

Este copiloto resolve os dois pontos:
1. **Fundamentação estrita (Zero Alucinação):** Recupera apenas as cláusulas relevantes da política vigente e instrui o modelo a decidir exclusivamente com base nelas — escalando automaticamente para o atendimento humano quando o caso não é coberto.
2. **Integração com sistemas internos:** Além de redigir a resposta pronta para o restaurante, devolve um **JSON tipado** pronto para alimentar filas de atendimento, CRMs ou fluxos automáticos de reembolso.

---

## ⚙️ Arquitetura e Como Funciona

O pipeline foi projetado para ser leve, auditável e livre de bancos vetoriais pesados:

1. **Chunking Semântico por Cláusulas (`motor_rag.py`):** O documento é fatiado respeitando a hierarquia de seções e cláusulas (`Cláusula X.Y`), preservando o contexto normativo completo (com *fallback* automático por parágrafos para PDFs genéricos).
2. **Busca Vetorial em Memória (`numpy`):** Os trechos são vetorizados via `gemini-embedding-001` (`task_type="RETRIEVAL_DOCUMENT"`) e armazenados em cache (`@st.cache_resource` indexado por hash MD5 do arquivo). A consulta do chamado (`RETRIEVAL_QUERY`) é comparada por similaridade de cosseno via produto escalar normalizado.
3. **Triagem Estruturada (`gemini-2.5-flash` + `Pydantic`):** Com `temperature=0.1` e `response_schema=TriagemChamado`, o modelo retorna obrigatoriamente o contrato abaixo:

```json
{
  "categoria": "cancelamento",
  "urgencia": "media",
  "elegivel_reembolso": true,
  "clausula_citada": "Cláusula 2.3",
  "precisa_atendimento_humano": false,
  "resposta_ao_parceiro": "Conforme a Cláusula 2.3, como o pedido foi marcado como pronto dentro do tempo informado e o atraso da entrega ultrapassou 30 minutos, você receberá 100% do valor dos itens, descontada apenas a comissão..."
}
