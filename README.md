# Vectra IT Knowledge Assistant

This assignment prototype answers staff questions from the Vectra IT knowledge base. `answer(question)` returns `answer`, `citations`, and `abstained`. OpenAI creates embeddings; LangChain LCEL runs chat through `prompt | model | parser`.

## Data and chunking

The 9 supplied Markdown articles were inspected. Each current article is stored whole as one chunk because all are short; this keeps the priority table in KB-003 and the error table in KB-007 intact. Seven current articles are ingested. KB-002 is skipped because it is deprecated and superseded by KB-007. KB-099 is skipped because it is unverified.

Each Pinecone document stores `doc_id`, `title`, `status`, `effective_date`, and `content` alongside its embedding. Search filters to `status=current`.

## Setup and run

Create a Pinecone index named `vectra-kb` with a dense vector field named `embedding`, dimension `1536`, and metric `cosine`. In `.env`, set `OPENAI_API_KEY`, `PINECONE_API_KEY`, and optionally `PINECONE_INDEX_NAME=vectra-kb`. Do not commit `.env` or share its keys.

In PowerShell, from this folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py ingest
python main.py
```

Enter an employee question at the prompt; enter `exit` to stop. Ingestion should report 7 current articles.

## Evaluation

Tested live against the five supplied questions and the KB edge cases. Checks covered source article, supported facts, citation, abstention, and unsafe or personal data in the answer.

- Account lock: 30-minute unlock; KB-001.
- VPN gateway: `zt-gw.vectralogistics.example`; KB-007, not deprecated KB-002.
- P2 SLA: 1-hour first response and 8-hour resolution; KB-003.
- Parking pass: abstains with no citation and suggests a ticket.
- iPhone built-in Mail: Outlook mobile with Intune; KB-004; contact details are not echoed.
- Linux and ZT-401 questions: supported guidance from KB-007.
- Slow laptop / antivirus: says not to disable antivirus; KB-006.
- Dock printer outage: P1 escalation and phone ext. 4357; KB-005 and KB-003.
- Ransomware: disconnect, call SecOps ext. 7333, and phone the P1 desk ext. 4357; KB-006 and KB-003.

The similarity floor is based on this small hand-built set and is not calibrated for general staff traffic.

## Before rollout to 3,000 staff

1. Re-index automatically when an article changes status or is superseded.
2. Never ingest unverified content such as KB-099.
3. Mask personal data before logging or sending questions to the model.
4. Build a larger test set from real tickets and run it after KB or prompt changes.
5. Ensure P1 answers include the phone instruction in KB-003.
6. Pilot with Service Desk agents before expanding to all staff.
