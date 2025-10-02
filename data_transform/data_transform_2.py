import pandas as pd
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq

TARGET = "C:/projects/big-data-tech-in-fin-manage/data/train_target.csv"
FILES = glob.glob("C:/projects/big-data-tech-in-fin-manage/data/train_data/*.pq")
OUT_PATH = "C:/projects/big-data-tech-in-fin-manage/data/train_aggregated.pq"

# загружаем таргеты
target = pd.read_csv(TARGET)
all_ids = target["id"].tolist()

batch_size = 10000
writer = None  # parquet writer

for i, chunk_ids in enumerate([all_ids[j:j+batch_size] for j in range(0, len(all_ids), batch_size)]):
    print(f"Обрабатываю чанки id {i*batch_size} – {(i+1)*batch_size}")

    agg_dict = {cid: {} for cid in chunk_ids}

    # собираем данные по всем parquet
    for f in FILES:
        df = pd.read_parquet(f)
        df = df[df["id"].isin(chunk_ids)]
        if df.empty:
            continue

        grouped = df.groupby("id").agg(lambda x: list(x))
        for cid, row in grouped.iterrows():
            for col, val in row.items():
                if col == "id":
                    continue
                if col not in agg_dict[cid]:
                    agg_dict[cid][col] = []
                agg_dict[cid][col].extend(val)

    # формируем датафрейм чанка
    chunk_df = pd.DataFrame.from_dict(agg_dict, orient="index")
    chunk_df.reset_index(inplace=True)
    chunk_df.rename(columns={"index": "id"}, inplace=True)

    # добавляем target
    chunk_df = target[target["id"].isin(chunk_ids)].merge(chunk_df, on="id", how="left")

    # --- запись в общий parquet через pyarrow ---
    table = pa.Table.from_pandas(chunk_df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT_PATH, table.schema)
    writer.write_table(table)

# закрываем writer
if writer:
    writer.close()

print("✅ Готово! Данные собраны в", OUT_PATH)
