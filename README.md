# CrewAI Multi-Agent FastAPI Backend

This repository provides a unified FastAPI backend that brings together 10 autonomous CrewAI projects originally built in Google Colab into a production-ready REST API.

Each project runs completely independently under its own dedicated endpoint: `/agent1` through `/agent10`.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Requirements & Installation](#requirements--installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Endpoints Reference](#api-endpoints-reference)
  - [/agent1 – Blog Post Crew](#agent1--blog-post-crew)
  - [/agent2 – Competitor Analysis Crew](#agent2--competitor-analysis-crew)
  - [/agent3 – Meeting Notes Crew](#agent3--meeting-notes-crew)
  - [/agent4 – Data Report Crew](#agent4--data-report-crew)
  - [/agent5 – Customer Support Flow](#agent5--customer-support-flow)
  - [/agent6 – Resume Screening Flow](#agent6--resume-screening-flow)
  - [/agent7 – Bug Triage Flow](#agent7--bug-triage-flow)
  - [/agent8 – Content Repurposing Flow](#agent8--content-repurposing-flow)
  - [/agent9 – Lead Qualification Flow](#agent9--lead-qualification-flow)
  - [/agent10 – Approval Report Flow](#agent10--approval-report-flow)
- [Interactive API Docs](#interactive-api-docs)

---

## 🚀 Project Overview

All original agents, task descriptions, prompts, LLM parameters, tools, sequential processes, and flow routers are preserved intact. Colab-specific input mechanisms (`input()`, `files.upload()`, `files.download()`, `userdata.get`) have been adapted to clean REST API endpoints accepting and returning standard JSON:

- **Request Format:**
  ```json
  {
    "input": "your input data here"
  }
  ```

- **Response Format:**
  ```json
  {
    "output": "actual project result here"
  }
  ```

Generated reports and output files (`blog_post.md`, `report.md`, `notes.md`, `support_ticket_resolution.md`, `results.csv`, `linkedin_post.txt`, `twitter_thread.txt`, `newsletter.txt`, `leads.csv`, `Final_Approved_Report.txt`) are created on disk as in the original notebooks.

---

## ⚙️ Requirements & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd CrewAI_Assignment
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   - **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔑 Configuration

Set your Google Gemini API key as an environment variable or in a `.env` file in the project root:

Create a file named `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Or set it in your terminal:
- **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="your_actual_gemini_api_key_here"
  ```
- **Linux / macOS:**
  ```bash
  export GEMINI_API_KEY="your_actual_gemini_api_key_here"
  ```

---

## 🖥️ Running the Server

Start the FastAPI application using Uvicorn:

```bash
uvicorn app:app --reload --port 8000
```
Or directly via Python:
```bash
python app.py
```

The server will be accessible at `http://127.0.0.1:8000`.

---

## 📚 API Endpoints Reference

### `/agent1` – Blog Post Crew
- **Description:** Researches a topic using DuckDuckGo, drafts an engaging 600-word post, and polishes grammar and clarity.
- **Agents:** Blog Researcher, Blog Writer, Blog Editor.
- **Output File:** `blog_post.md`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent1 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Artificial Intelligence in Healthcare\"}"
  ```

---

### `/agent2` – Competitor Analysis Crew
- **Description:** Scrapes competitor websites, compares pricing tiers, searches DuckDuckGo for news, and generates 4-column SWOT tables (`| Strengths | Weaknesses | Opportunities | Threats |`).
- **Agents:** Site Researcher, Pricing Analyst, News Analyst, Strategist, Manager.
- **Output File:** `report.md`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent2 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"https://bitly.com, https://rebrandly.com, https://bl.ink\"}"
  ```

---

### `/agent3` – Meeting Notes Crew
- **Description:** Analyzes meeting transcripts, summarizes in exactly 5 lines, extracts action items with owners and due dates, logs confirmed decisions, and formats professional notes.
- **Agent:** Meeting Notes Formatter.
- **Output File:** `notes.md`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent3 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Alice: Welcome everyone. We must decide our backend framework today.\nBob: I propose FastAPI due to automatic OpenAPI docs and high performance.\nCharlie: Agreed. Bob will implement the base structure by Friday.\nAlice: Perfect, decision confirmed.\"}"
  ```

---

### `/agent4` – Data Report Crew
- **Description:** Analyzes sales CSV data using Pandas, computes statistical metrics, derives top 3 business insights, verifies all figures, and produces a structured sales report.
- **Agents:** Data Analyst, Insight Writer, Sales Report Reviewer.
- **Output File:** `sales_report.md`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent4 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"product,region,sales,quantity\nWidget A,North,1200,10\nWidget B,South,3400,25\nWidget A,East,2100,18\nWidget C,West,800,5\"}"
  ```

---

### `/agent5` – Customer Support Flow
- **Description:** Event-driven Flow that receives support tickets, classifies them (Billing, Account Access, Technical Issue, Feature, Other), applies routing, generates a resolution plan, and saves markdown documentation.
- **Flow:** `CustomerSupportFlow`
- **Output File:** `support_ticket_resolution.md`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent5 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Money was deducted from my bank account, but my order status shows failed.\"}"
  ```

---

### `/agent6` – Resume Screening Flow
- **Description:** Flow that screens applicant resumes against a job description, assigns scores (0–10), routes to shortlist (>= 7) or rejection (< 7), generates suitability summaries or polite rejection emails, and logs structured CSV results.
- **Flow:** `ResumeScreeningFlow`
- **Output File:** `results.csv`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent6 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"JOB DESCRIPTION:\nSenior Python Engineer with experience in FastAPI and multi-agent AI frameworks.\n\nRESUME:\nCandidate Name: Jane Smith\nSkills: Python, FastAPI, CrewAI, Docker, PostgreSQL\nExperience: 4 years designing REST APIs and AI agent workflows.\"}"
  ```

---

### `/agent7` – Bug Triage Flow
- **Description:** Flow that classifies incoming issue reports into `bug`, `feature`, or `question`, and routes them to dedicated sub-crews (Bug Investigator, Feature Analyst, or Question Answering Specialist).
- **Flow:** `BugTriageFlow`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent7 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Title: Database timeout on login\nDescription: Users experience 504 Gateway Timeout when submitting the login form under heavy traffic.\"}"
  ```

---

### `/agent8` – Content Repurposing Flow
- **Description:** Flow that reads an article (via text or URL), extracts core points, and concurrently spawns three sub-crews to generate a 150-word LinkedIn post, a 6-tweet Twitter thread, and an executive newsletter.
- **Flow:** `ContentRepurposingFlow`
- **Output Files:** `linkedin_post.txt`, `twitter_thread.txt`, `newsletter.txt`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent8 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Autonomous AI agents are transforming enterprise operations. By combining LLMs with external tools and structured workflows, agents can independently research, analyze data, and generate deliverables with minimal supervision.\"}"
  ```

---

### `/agent9` – Lead Qualification Flow
- **Description:** Flow that enriches business leads using website scraping, scores them from 0–100, categorizes them (`hot`, `warm`, `cold`), crafts personalized sales outreach or nurturing materials, and appends to a CSV ledger.
- **Flow:** `LeadQualificationFlow`
- **Output File:** `leads.csv`
- **Example Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/agent9 \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Acme Cloud Solutions, https://example.com\"}"
  ```

---

### `/agent10` – Approval Report Flow
- **Description:** Human-in-the-loop Flow that conducts multi-agent research, drafts an in-depth report, presents it for human evaluation (`@human_feedback`), and dynamically revises or approves and publishes the final report.
- **Flow:** `ApprovalReportFlow`
- **Output File:** `Final_Approved_Report.txt`

#### How Human-in-the-Loop Works:
1. **Initial Draft Generation:**
   ```bash
   curl -X POST http://127.0.0.1:8000/agent10 \
     -H "Content-Type: application/json" \
     -d "{\"input\": \"Topic: Quantum Computing in Cryptography | Audience: Cybersecurity Engineers\"}"
   ```
   *Response:* Returns the generated draft report with status `DRAFT GENERATED — AWAITING HUMAN FEEDBACK`.

2. **Requesting Revisions (Human Rejection):**
   ```bash
   curl -X POST http://127.0.0.1:8000/agent10 \
     -H "Content-Type: application/json" \
     -d "{\"input\": \"Topic: Quantum Computing in Cryptography | Feedback: REVISE: Focus more on post-quantum encryption standards like NIST lattice-based algorithms.\"}"
   ```
   *Response:* Executes the `Report Revision Specialist` agent and returns the updated draft.

3. **Final Approval (Human Sign-off):**
   ```bash
   curl -X POST http://127.0.0.1:8000/agent10 \
     -H "Content-Type: application/json" \
     -d "{\"input\": \"Topic: Quantum Computing in Cryptography | Feedback: APPROVE\"}"
   ```
   *Response:* Publishes the report, saves `Final_Approved_Report.txt`, and returns status `APPROVED AND PUBLISHED`.

---

## 🌐 Interactive API Docs

FastAPI automatically generates interactive Swagger documentation. Once the server is running, open your browser to:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
