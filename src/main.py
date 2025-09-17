# src/main.py
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
from typing import Optional, List

# ---- CONFIG ----
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_BASE = Path(__file__).resolve().parents[1] / "out"
MONTH_DIR = OUT_BASE / "months"

# Take ALL .csv files in data/
def list_input_files() -> List[Path]:
    return sorted((p for p in DATA_DIR.glob("*.csv") if p.is_file()),
                  key=lambda p: p.name.lower())
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# ---------------- Helpers ----------------

def parse_date_any(s: str) -> Optional[datetime]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() == "nan":
        return None

    m = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$", t)  # yyyy-mm-dd
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d)
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", t)  # mm/dd/yyyy
    if m:
        mo, d, y = m.groups()
        y = int(y)
        if y < 100:
            y += 2000
        try:
            return datetime(int(y), int(mo), int(d))
        except Exception:
            return None

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,}\.?)\s+(\d{4})\b", t)  # 13 Sep. 2025
    if m:
        d = int(m.group(1))
        mon = m.group(2).lower().rstrip(".")
        y = int(m.group(3))
        mo = MONTHS.get(mon)
        if mo:
            try:
                return datetime(y, mo, d)
            except Exception:
                return None

    m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", t)  # 2025/9/3
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d)
        except Exception:
            return None

    return None


def money_to_float(s) -> Optional[float]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    txt = str(s).strip()
    sign = -1.0 if re.search(r"(^|[^0-9])-\s*\$?\s*\d", txt) else 1.0
    num = re.search(r"\d+(?:,\d{3})*(?:\.\d{1,2})?", txt)
    return sign * float(num.group(0).replace(",", "")) if num else None


def textiest_col_idx(df: pd.DataFrame) -> int:
    best_idx, best_score = 0, -1
    for i in range(df.shape[1]):
        series = df.iloc[:, i].dropna().astype(str)
        score = series.map(lambda s: len(s) > 2 and not re.fullmatch(r"[\$,\-\d\. ]+", s)).sum()
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def moneylike_col_idx(df: pd.DataFrame) -> int:
    money_re = re.compile(r"^\s*-?\$?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?\s*$")
    best_idx, best_score = 0, -1
    for i in range(df.shape[1]):
        series = df.iloc[:, i].dropna().astype(str)
        score = series.map(lambda s: bool(money_re.match(s))).sum()
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx

# ---------------- Parsers ----------------

def parse_amex_summary_with_headers(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False)

    # Find header row where first two columns are "Date" and "Date Processed"
    header_row = None
    for i in range(min(150, len(raw))):
        c0 = str(raw.iloc[i, 0]).strip().lower()
        c1 = str(raw.iloc[i, 1]).strip().lower() if raw.shape[1] > 1 else ""
        if c0 == "date" and "date processed" in c1:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Amex: could not locate header row")

    header = list(raw.iloc[header_row])
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header

    if "Date" not in df.columns:
        raise ValueError("Amex: 'Date' column not found after header normalization")

    if "Description" not in df.columns:
        for c in list(df.columns):
            if str(c).strip().lower() == "description":
                df = df.rename(columns={c: "Description"})
                break
        if "Description" not in df.columns:
            df = df.rename(columns={df.columns[2]: "Description"})

    amt_col = None
    for name in ["Amount", "Charge Amount", "CAD$", "Amount (CAD)"]:
        if name in df.columns:
            amt_col = name
            break
    if amt_col is None:
        amt_col = df.columns[moneylike_col_idx(df)]

    out = pd.DataFrame(
        {
            "date": df["Date"].map(parse_date_any),
            "name": df["Description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "price": df[amt_col].map(money_to_float),
            "account": "Amex",
            "source": path.name,
        }
    ).dropna(subset=["date", "name", "price"])
    return out


def parse_visa_headerless(path: Path) -> Optional[pd.DataFrame]:
    df = pd.read_csv(path, header=None, dtype=str)
    if not re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", str(df.iloc[0, 0])):
        return None

    numeric_cols = []
    for c in range(2, df.shape[1]):
        series = df[c].astype(str)
        score = series.str.contains(r"\d").sum()
        numeric_cols.append((c, score))
    numeric_cols.sort(key=lambda x: x[1], reverse=True)
    debit_col = numeric_cols[0][0] if len(numeric_cols) >= 1 else None
    credit_col = numeric_cols[1][0] if len(numeric_cols) >= 2 else None

    def m2f(x):
        if x is None:
            return 0.0
        t = str(x).replace(",", "").strip()
        if t == "" or t.lower() == "nan":
            return 0.0
        m = re.search(r"-?\d+(?:\.\d{1,2})?$", t)
        return float(m.group(0)) if m else 0.0

    debit = df[debit_col].map(m2f) if debit_col is not None else 0.0
    credit = df[credit_col].map(m2f) if credit_col is not None else 0.0
    amount = credit - debit  # credits positive, debits negative

    out = pd.DataFrame(
        {
            "date": df[0].map(parse_date_any),
            "name": df[1].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "price": amount,
            "account": "Visa",
            "source": path.name,
        }
    ).dropna(subset=["date", "name"])
    return out


def parse_generic_named(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    lower = {str(c).lower(): c for c in df.columns}

    date_col = next(
        (lower[k] for k in ["date", "transaction date", "trans. date", "posted date", "post date", "purchase date", "transactiondate"] if k in lower),
        max(df.columns, key=lambda c: df[c].astype(str).map(lambda v: parse_date_any(v) is not None).sum()),
    )
    desc_col = next(
        (lower[k] for k in ["description", "merchant", "details", "name", "transaction", "transaction description", "narrative", "memo"] if k in lower),
        df.columns[textiest_col_idx(df)],
    )

    amt_col = next((lower[k] for k in ["amount", "amount ($)", "transaction amount", "cad$", "amount (cad)", "value"] if k in lower), None)
    debit_col = next((lower[k] for k in ["debit", "debits", "withdrawal", "debit amount"] if k in lower), None)
    credit_col = next((lower[k] for k in ["credit", "credits", "deposit", "credit amount"] if k in lower), None)

    if amt_col:
        amount = df[amt_col].map(money_to_float)
    elif debit_col or credit_col:
        deb = df[debit_col].map(money_to_float) if debit_col else 0.0
        cre = df[credit_col].map(money_to_float) if credit_col else 0.0
        amount = cre - deb
    else:
        amount = df.iloc[:, moneylike_col_idx(df)].map(money_to_float)

    out = pd.DataFrame(
        {
            "date": df[date_col].map(parse_date_any),
            "name": df[desc_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
            "price": amount,
            "account": "Visa/Other",
            "source": path.name,
        }
    ).dropna(subset=["date", "name", "price"])
    return out


def parse_one(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if "summary" in name:
        return parse_amex_summary_with_headers(path)
    df = parse_visa_headerless(path)
    if df is not None:
        return df
    return parse_generic_named(path)

# ---------------- Main ----------------

def to_4cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 4 requested columns in the desired order/names."""
    out = pd.DataFrame()
    out["number day"] = df["date"].dt.day.astype(int)
    out["description"] = df["name"].astype(str)
    out["price"] = df["price"].astype(float)
    out["fulldate"] = df["date"].dt.strftime("%Y-%m-%d")
    return out

def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    MONTH_DIR.mkdir(parents=True, exist_ok=True)

    input_files = list_input_files()
    if not input_files:
        print(f"No files found matching report*.csv in {DATA_DIR}")
        return

    parsed_frames = []
    for in_path in input_files:
        try:
            dfo = parse_one(in_path)  # columns: date, name, price, account, source
            # write per-file (4 columns)
            out_file = OUT_BASE / f"formatted_{in_path.stem}.csv"
            to_4cols(dfo).to_csv(out_file, index=False)
            parsed_frames.append(dfo)
            print(f"✅ {in_path.name} → {out_file.name} ({len(dfo)} rows)")
        except Exception as e:
            print(f"❌ {in_path.name}: {e}")

    if not parsed_frames:
        print("No files parsed.")
        return

    combined_raw = pd.concat(parsed_frames, ignore_index=True)

    # Combined output (4 columns)
    combined_4 = to_4cols(combined_raw)
    combined_file = OUT_BASE / "combined_all.csv"
    combined_4.to_csv(combined_file, index=False)
    print(f"📦 Combined → {combined_file}")

    # Monthly splits (4 columns), sorted by date
    combined_raw["month"] = combined_raw["date"].dt.strftime("%Y-%m")
    for month, dfm in combined_raw.groupby("month", sort=True):
        dfm_sorted = dfm.sort_values("date")
        mp = MONTH_DIR / f"{month}.csv"
        to_4cols(dfm_sorted).to_csv(mp, index=False)
        print(f"🗓  {month} → {mp}")

    # quick console summary (unchanged logic)
    summary = combined_raw.groupby("month").agg(transactions=("price", "size"),
                                                spend_total=("price", "sum")).reset_index()
    print("\nSummary by month:")
    for _, r in summary.iterrows():
        print(f"  {r['month']}: {int(r['transactions'])} txns, total {r['spend_total']:.2f}")

if __name__ == "__main__":
    main()
