import json, os, re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import DenseVectorQuery, Pinecone
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
ai = OpenAI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.index(os.getenv("PINECONE_INDEX_NAME", "vectra-kb"))
namespace = "vectra-kb"


prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the supplied current KB articles. Do not invent information. If unsupported, say so and suggest raising a ticket. For P1 incidents, always tell the employee to phone the 24x7 service desk at extension 4357 in addition to raising the ticket. Return only JSON with exactly these keys: answer (string), citations (list of KB IDs), abstained (boolean)."),
    ("human", "Question: {question}\n\nKB articles:\n{sources}"),
])
model = ChatOpenAI(model="gpt-6-luna", use_responses_api=True)
parser = JsonOutputParser()

chain = prompt | model | parser


def read_kb():
    articles = []
    kb_dir = Path(__file__).parent / "vectra_kb_articles" / "kb"

    for path in kb_dir.glob("*.md"):
        raw = path.read_text(encoding="utf-8")
        header, content = raw.split("---", 2)[1:]
        metadata = dict(line.split(":", 1) for line in header.strip().splitlines())
        metadata = {key.strip(): value.strip() for key, value in metadata.items()}
        if metadata["status"] != "current":
            continue
        articles.append({
            "id": metadata["doc_id"],
            "title": metadata["title"],
            "status": metadata["status"],
            "effective_date": metadata["effective_date"],
            "content": content.strip(),
        })
    return articles


def embed(text):
    return ai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding


def ingest():
    documents = [{
        "_id": article["id"],
        "embedding": embed(article["title"] + "\n" + article["content"]),
        "doc_id": article["id"],
        "title": article["title"],
        "status": article["status"],
        "effective_date": article["effective_date"],
        "content": article["content"],
    } for article in read_kb()]
    index.documents.upsert(namespace=namespace, documents=documents)
    print("Ingested current articles:", len(documents))


def answer(question):
    question = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "[EMAIL]", question)
    question = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[PHONE]", question)
    result = index.documents.search(
        namespace=namespace,
        top_k=4,
        score_by=[DenseVectorQuery(field="embedding", values=embed(question))],
        filter={"status": {"$eq": "current"}},
        include_fields=["doc_id", "title", "content"],
    )
    matches = result.matches
    if not matches or matches[0].score < 0.35:
        return {"answer": "The KB does not cover this. Please raise a ticket.",
                "citations": [], "abstained": True}

    sources = [f"[{m.get('doc_id')}] {m.get('title')}\n{m.get('content')}" for m in matches]
    allowed_ids = {m.get("doc_id") for m in matches}
    try:
        data = chain.invoke({"question": question, "sources": "\n\n".join(sources)})
        citations = data["citations"]
        if data["abstained"] or not citations or any(c not in allowed_ids for c in citations):
            raise ValueError("Unsupported answer or citation")
        return {"answer": data["answer"], "citations": citations, "abstained": False}
    except Exception:
        return {"answer": "The KB does not cover this. Please raise a ticket.",
                "citations": [], "abstained": True}

if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "ingest":
        ingest()
    else:
        while True:
            question = input("Question (or exit): ")
            if question.lower() == "exit":
                break
            print(json.dumps(answer(question), indent=2))
