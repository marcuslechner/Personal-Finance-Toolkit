# Personal Finance Toolkit

A Python project to parse credit card statements, format transactions, and automatically categorize expenses.  
The goal is to make sense of monthly spending with minimal effort — and eventually visualize habits and connect to AI tools for insights.

---

## 🚀 Features (Planned)
- **Parse Statements** – Read CSV/Excel statements and normalize into a standard format.
- **Auto-Categorize** – Match merchants & keywords from a `categories.json` config file.
- **Summarize Spending** – Generate quick monthly breakdowns by category.
- **Future Add-ons**:
  - Visual dashboards of spending habits.
  - GPT-powered category suggestions.
  - Tools for budgeting and forecasting.

---

## 📦 Project Setup

```bash
# 1. Clone this repo
git clone https://github.com/your-username/personal-finance-toolkit
cd personal-finance-toolkit

# 2. Create & activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt  # (or manually install pandas, typer, etc.)
