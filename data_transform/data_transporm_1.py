import pandas as pd
import glob, gc, os

# Пути
files = glob.glob("C:/projects/big-data-tech-in-fin-manage/data/train_data/*.pq")
tmp_dir = "C:/projects/big-data-tech-in-fin-manage/tmp_agg"
os.makedirs(tmp_dir, exist_ok=True)

# Этап 1. Агрегация по файлам
for i, f in enumerate(files):
    print(f"Обрабатываю {f} ...")
    df = pd.read_parquet(f)
    
    # группируем: id → списки значений
    df = df.groupby("id").agg(lambda x: list(x))
    
    # сохраняем промежуточный parquet
    df.to_parquet(f"{tmp_dir}/agg_{i}.pq")
    
    # чистим память
    del df
    gc.collect()

# Этап 2. Объединение результатов
print("Объединяю промежуточные файлы...")
agg_files = glob.glob(f"{tmp_dir}/agg_*.pq")

agg_list = []
for f in agg_files:
    df = pd.read_parquet(f)
    agg_list.append(df)
    del df
    gc.collect()

agg_data = pd.concat(agg_list).groupby("id").agg(lambda x: sum(x, []))

# Этап 3. Джойн с таргетами
print("Джойн с таргетами...")
target = pd.read_csv("C:/projects/big-data-tech-in-fin-manage/data/train_target.csv")
final = target.merge(agg_data, on="id", how="left")

# Этап 4. Сохраняем итог
final.to_parquet("C:/projects/big-data-tech-in-fin-manage/data/train_aggregated.pq")
print("✅ Готово!")
