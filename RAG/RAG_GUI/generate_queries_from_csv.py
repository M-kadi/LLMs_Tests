import pandas as pd
from pathlib import Path

# -------- CONFIG --------
CSV_PATH = Path("D:/LLM/LLM_Tests/RAG_GUI/rag_data/my_csvs/docs3/file2.csv")   # change if needed
OUTPUT_TXT = Path("D:/LLM/LLM_Tests/RAG_GUI/rag_data/my_csvs/generated_queries.txt")

FIELDS = [
    "ClaimType",
    "InitialEncounterStartType",
    "Initial_EncounterEndType",
    "Initial_EncounterType",
    "Initial_PayerId",
    "Initial_PlanName",
    "Initial_Claimstatus",
    "Initial_PrincipalDiagnosisCode",
    "Initial_SecondaryDiagnosisCode",
]

LABEL_COL = "Initial_ActivityDenailCode"
# ------------------------

def clean_value(v):
    if pd.isna(v):
        return "No Data"
    return str(v).strip()

def main():
    df = pd.read_csv(CSV_PATH)

    blocks = []
    i = 0
    for _, row in df.iterrows():
        lines = []

        # input fields
        for field in FIELDS:
            value = clean_value(row.get(field))
            lines.append(f"{field}={value}")

        # label
        label = clean_value(row.get(LABEL_COL))
        lines.append(f"The Initial_ActivityDenailCode is {label}")
        i = i + 1
        # block separator
        lines.append(f"#{i}")

        blocks.append("\n".join(lines))

    OUTPUT_TXT.write_text("\n\n".join(blocks), encoding="utf-8")

    print(f"✅ Generated {len(blocks)} queries")
    print(f"📄 Saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
