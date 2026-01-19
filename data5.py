import os
import pandas as pd
import re
from typing import List

from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from flask import Flask, request, jsonify, render_template


# CONFIG

load_dotenv()

CHROMA_PATH = "chroma_db"
os.makedirs(CHROMA_PATH, exist_ok=True)

app = Flask(__name__)

TRADES_FILE = r"D:\project_krish\propety_loop\trades.csv"
HOLDINGS_FILE = r"D:\project_krish\propety_loop\holdings.csv"

NO_ANSWER_MSG = "Sorry, I cannot find the answer from the available data."


# LOAD DATAFRAMES

trades_df = pd.read_csv(TRADES_FILE)
holdings_df = pd.read_csv(HOLDINGS_FILE)

holdings_df["AsOfDate"] = pd.to_datetime(holdings_df["AsOfDate"], errors="coerce")
trades_df["TradeDate"] = pd.to_datetime(trades_df["TradeDate"], errors="coerce")


# COLUMN METADATA

COLUMN_METADATA = {
    "pnl": ["PL_DTD", "PL_MTD", "PL_QTD", "PL_YTD"],
    "market_value": ["MV_Base", "MV_Local"],
    "price": ["Price", "StartPrice"],
    "quantity": ["Qty", "StartQty", "Quantity"],
    "portfolio": ["PortfolioName"],
    "strategy": [
        "StrategyRefShortName",
        "Strategy1RefShortName",
        "Strategy2RefShortName"
    ]
}


# DATE RANGE PARSER

def parse_date_range(question: str):
    matches = re.findall(r"\d{4}-\d{2}-\d{2}", question)
    if len(matches) == 2:
        return pd.to_datetime(matches[0]), pd.to_datetime(matches[1])
    return None, None


# NUMERIC QUERY HANDLER

def handle_numeric_query(question: str):
    q = question.lower()

    # ───────── COUNT QUERIES (HOLDINGS / TRADES) ─────────
    if "number of" in q or "how many" in q or "count" in q:

        # Detect fund / portfolio
        fund_name = None
        for p in holdings_df["PortfolioName"].dropna().unique():
            if p.lower() in q:
                fund_name = p
                break

        # ---- HOLDINGS COUNT ----
        if "holding" in q:
            df = holdings_df
            if fund_name:
                df = df[df["PortfolioName"] == fund_name]
                return f"Total number of holdings for {fund_name} is {len(df)}"
            return f"Total number of holdings is {len(df)}"

        # ---- TRADES COUNT ----
        if "trade" in q:
            df = trades_df
            if fund_name:
                df = df[df["PortfolioName"] == fund_name]
                return f"Total number of trades for {fund_name} is {len(df)}"
            return f"Total number of trades is {len(df)}"

    # ───────── HOLDINGS NUMERIC LOGIC ─────────
    df = holdings_df.copy()

    start_date, end_date = parse_date_range(q)
    if start_date and end_date:
        df = df[(df["AsOfDate"] >= start_date) & (df["AsOfDate"] <= end_date)]

    for p in df["PortfolioName"].dropna().unique():
        if p.lower() in q:
            df = df[df["PortfolioName"] == p]
            break

    for col in COLUMN_METADATA["strategy"]:
        if col in df.columns:
            for s in df[col].dropna().unique():
                if s.lower() in q:
                    df = df[df[col] == s]
                    break

    if "p&l" in q or "profit" in q or "loss" in q:
        for col in COLUMN_METADATA["pnl"]:
            if col.replace("PL_", "").lower() in q:
                return f"{col} is {df[col].sum():,.2f}"
        return f"Total YTD P&L is {df['PL_YTD'].sum():,.2f}"

    if "market value" in q:
        return f"Total market value is {df['MV_Base'].sum():,.2f}"

    if "average price" in q:
        return f"Average price is {df['Price'].mean():.2f}"

    # ───────── TRADES NUMERIC LOGIC ─────────
    trade_df = trades_df.copy()

    if start_date and end_date:
        trade_df = trade_df[
            (trade_df["TradeDate"] >= start_date) &
            (trade_df["TradeDate"] <= end_date)
        ]

    if "total traded quantity" in q:
        return f"Total traded quantity is {trade_df['Quantity'].sum():,.2f}"

    return None


# LOAD DOCUMENTS FOR RAG

def load_data() -> List[Document]:
    docs = []

    for _, row in trades_df.iterrows():
        docs.append(Document(
            page_content=f"TRADE DATA: {row.to_dict()}",
            metadata={"source": "trades"}
        ))

    for _, row in holdings_df.iterrows():
        docs.append(Document(
            page_content=f"HOLDING DATA: {row.to_dict()}",
            metadata={"source": "holdings"}
        ))

    return docs


# VECTOR STORE

_vector_store = None

def get_vector_store():
    global _vector_store

    if _vector_store:
        return _vector_store

    embeddings = OpenAIEmbeddings()

    if os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
        _vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        return _vector_store

    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    split_docs = []

    for doc in load_data():
        for chunk in splitter.split_text(doc.page_content):
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    _vector_store = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return _vector_store


# RAG CHAIN

def get_rag_chain():
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """Answer ONLY using the context.
If the answer is not present, reply exactly:
"Sorry, I cannot find the answer from the available data."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json() or {}).get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question."})

    # STEP-1: pandas numeric / count logic
    numeric_answer = handle_numeric_query(question)
    if numeric_answer:
        return jsonify({"answer": numeric_answer})

    # STEP-2: strict RAG fallback
    vectordb = get_vector_store()
    if not vectordb.similarity_search(question, k=3):
        return jsonify({"answer": NO_ANSWER_MSG})

    chain = get_rag_chain()
    response = chain.invoke(question)
    answer = response.content.strip()

    if not answer or "cannot find" in answer.lower():
        return jsonify({"answer": NO_ANSWER_MSG})

    return jsonify({"answer": answer})


# RUN

if __name__ == "__main__":
    print("Starting CSV Portfolio Assistant (Counts + Analytics Enabled)")
    get_vector_store()
    app.run(host="0.0.0.0", port=5000, debug=True)
