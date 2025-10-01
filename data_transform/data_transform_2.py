import os
import gc
import glob
import pandas as pd

# Пути
files = glob.glob("C:/projects/big-data-tech-in-fin-manage/tmp_agg/*.pq")
OUT_PATH = r"C:/projects/big-data-tech-in-fin-manage/data/agg_merged_only.parquet"
FINAL_WITH_TARGET = r"C:/projects/big-data-tech-in-fin-manage/data/train_aggregated.pq"
TARGET_CSV = r"C:/projects/big-data-tech-in-fin-manage/data/train_target.csv"

def list_cat(a, b):
    """Безопасная конкатенация списков (NaN -> [])."""
    la = a if isinstance(a, list) else []
    lb = b if isinstance(b, list) else []
    return la + lb

def ensure_index(df):
    """Гарантируем индекс по id (в твоих файлах, судя по скрину, id уже индекс)."""
    if "id" in df.columns:
        df = df.set_index("id")
    return df

files = sorted(files)
if not files:
    raise RuntimeError("Не найдены промежуточные parquet!")

base = None
cols = None
checkpoint_every = 3

for i, path in enumerate(files):
    print(f"Обрабатываю файл {i+1}/{len(files)}: {os.path.basename(path)}")
    df = pd.read_parquet(path)
    df = ensure_index(df)

    if base is None:
        base = df
        cols = list(base.columns)
        for c in cols:
            if base[c].dtype != "object":
                base[c] = base[c].astype("object")
        del df
        gc.collect()
        continue

    base, df = base.align(df, join="outer", axis=0)

    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series([[]] * len(df), index=df.index, dtype="object")
        if c not in base.columns:
            base[c] = pd.Series([[]] * len(base), index=base.index, dtype="object")
        base[c] = base[c].combine(df[c], list_cat)

    del df
    gc.collect()

    if (i + 1) % checkpoint_every == 0:
        tmp_out = OUT_PATH + ".checkpoint"
        base.to_parquet(tmp_out)
        # перечитываем без set_index("id") → индекс сохраняется как есть
        base = pd.read_parquet(tmp_out)
        base.index.name = "id"
        os.remove(tmp_out)
        gc.collect()

# итог: сохраняем объединённый агрегат
base.to_parquet(OUT_PATH)

# --- добавляем таргеты ---
print("Джойн с таргетами...")
target = pd.read_csv(TARGET_CSV)
if "id" not in base.columns:
    base = base.reset_index()
final = target.merge(base, on="id", how="left")

final.to_parquet(FINAL_WITH_TARGET, index=False)

print("✅ Готово!")
print(f"- Объединённый агрегат без target: {OUT_PATH}")
print(f"- Финальный датасет с target:      {FINAL_WITH_TARGET}")
