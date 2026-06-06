

# =========================
# 🧹 Step 4: Standardize Column Names
# =========================

glucose_cpep_insulin.columns = cpep_insulin.columns.str.upper()
chol.columns = chol.columns.str.upper()
fasting.columns = fasting.columns.str.upper()


# =========================
# 🔗 Step 5: Merge ALL columns (FULL OUTER JOIN)
# =========================

df = glucose.merge(glucose_cpep_insulin, on="SEQN", how="outer") \
            .merge(chol, on="SEQN", how="outer") \
            .merge(fasting, on="SEQN", how="outer")


# =========================
# ⚠️ Step 6: Keep Missing Values AS-IS (NO CLEANING)
# =========================
# We DO NOT replace "." or 7/9 because user wants raw data preserved


# =========================
# 👀 Step 7: Preview
# =========================
print("Final Merged Dataset Preview:")
display(df.head(10))

print("\nShape:", df.shape)


# =========================
# 📊 Step 8: Basic Validation
# =========================
print("\nMissing values (top 20 columns):")
print(df.isnull().sum().sort_values(ascending=False).head(20))


# =========================
# 💾 Step 9: Save Output
# =========================
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "nhanes_labs_full_all_columns.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:")
print(OUTPUT_PATH)
'''