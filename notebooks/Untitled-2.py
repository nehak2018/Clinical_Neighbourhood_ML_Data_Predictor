
# =========================
# 📈 Step 8: Count how many datasets each SEQN appears in
# =========================
seqn_presence_cols = [
    "SEQN_present_glucose_cpep_insulin",
    "SEQN_present_chol",
    "SEQN_present_fasting"
]

df["SEQN_source_count"] = df[seqn_presence_cols].sum(axis=1)

# =========================
# 👀 Step 9: Preview SEQN Validation
# =========================
preview_cols = [
    "SEQN",
    "SEQN_GLUCOSE_CPEP_INSULIN",
    "SEQN_CHOL",
    "SEQN_FASTING",
    "SEQN_present_glucose_cpep_insulin",
    "SEQN_present_chol",
    "SEQN_present_fasting",
    "SEQN_source_count"
]

print("\n🔍 SEQN Validation Preview:")
display(df[preview_cols].head(20))



# =========================
# 📊 Step 10: Missing SEQN Summary
# =========================
print("\n📊 Missing SEQN by Source:")
print("Glucose/C-peptide/Insulin missing:", df["SEQN_GLUCOSE_CPEP_INSULIN"].isna().sum())
print("Cholesterol missing:", df["SEQN_CHOL"].isna().sum())
print("Fasting missing:", df["SEQN_FASTING"].isna().sum())


# =========================
# 📊 Step 11: Distribution
# =========================
print("\n📊 Number of datasets each participant appears in:")
print(df["SEQN_source_count"].value_counts().sort_index())


# =========================
# ⚠️ Step 12: Show incomplete merges
# =========================
missing_any = df[df["SEQN_source_count"] < 3]

print("\n⚠️ Participants missing from at least one dataset:")
display(
    missing_any[
        [
            "SEQN",
            "SEQN_GLUCOSE_CPEP_INSULIN",
            "SEQN_CHOL",
            "SEQN_FASTING",
            "SEQN_source_count"
        ]
    ].head(20)
)


# =========================
# 👀 Step 13: Full Dataset Preview
# =========================
print("\n📋 Final merged dataset preview:")
display(df.head(10))
print("\nFinal Shape:", df.shape)
