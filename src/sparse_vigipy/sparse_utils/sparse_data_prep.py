from itertools import chain, combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from .Container import Container


def _build_sparse_indicator(df, col_or_cols, weights=None, dtype=np.int8):
    """Build a CSR sparse matrix and corresponding column names from one or more
    label columns (supports lists of columns)."""
    cols = [col_or_cols] if isinstance(col_or_cols, str) else list(col_or_cols)
    n_rows = len(df)
    w = np.ones(n_rows, dtype=dtype) if weights is None else np.asarray(weights, dtype=dtype)

    row_parts = []
    val_parts = []
    name_parts = []

    for c in cols:
        series = df[c]
        valid_mask = series.notna() & (series.astype(str) != "")
        valid_rows = np.flatnonzero(valid_mask.to_numpy())
        if len(valid_rows) == 0:
            continue

        cat = series.iloc[valid_rows].astype("category")
        row_parts.append(valid_rows)
        val_parts.append(w[valid_rows])
        name_parts.append((cat.cat.codes.to_numpy(), cat.cat.categories.tolist()))

    if not name_parts:
        return csr_matrix((n_rows, 0), dtype=dtype), []

    # Unify unique categories across all provided columns
    all_categories = sorted({cat_name for _, cats in name_parts for cat_name in cats})
    cat_to_idx = {name: idx for idx, name in enumerate(all_categories)}

    col_indices = []
    for codes, cats in name_parts:
        local_map = np.array([cat_to_idx[name] for name in cats], dtype=np.int32)
        col_indices.append(local_map[codes])

    rows_concat = np.concatenate(row_parts)
    cols_concat = np.concatenate(col_indices)
    vals_concat = np.concatenate(val_parts)

    mat = csr_matrix(
        (vals_concat, (rows_concat, cols_concat)),
        shape=(n_rows, len(all_categories)),
        dtype=dtype,
    )
    return mat, all_categories


def convert(
    data_frame,
    margin_threshold=1,
    product_label="name",
    count_label="count",
    ae_label="AE",
):
    """Convert an event DataFrame into a VIGIPY Container using a CSR sparse matrix."""
    # 1. Parallel aggregation via Polars (avoids dense pivot)
    if not isinstance(data_frame, pl.DataFrame):
        lf = pl.from_pandas(data_frame).lazy()
    else:
        lf = data_frame.lazy()

    agg_df = (
        lf.select([product_label, ae_label, count_label])
        .drop_nulls([product_label, ae_label])
        .group_by([product_label, ae_label])
        .agg(pl.col(count_label).sum())
        .filter(pl.col(count_label) >= margin_threshold)
        .collect()
    )

    # 2. Fast categorical encoding and direct CSR matrix construction
    prod_cat = agg_df[product_label].to_pandas().astype("category")
    ae_cat = agg_df[ae_label].to_pandas().astype("category")

    product_names = prod_cat.cat.categories.to_numpy()
    ae_names = ae_cat.cat.categories.to_numpy()

    row_idx = prod_cat.cat.codes.to_numpy()
    col_idx = ae_cat.cat.codes.to_numpy()
    counts = agg_df[count_label].to_numpy().astype(np.float64)

    data_cont = csr_matrix(
        (counts, (row_idx, col_idx)),
        shape=(len(product_names), len(ae_names)),
        dtype=np.float64,
    )

    # 3. Compute marginal sums directly on the sparse matrix
    row_sums = np.asarray(data_cont.sum(axis=1)).ravel()
    col_sums = np.asarray(data_cont.sum(axis=0)).ravel()

    # 4. Vectorized reconstruction of data_df on non-zero entries only
    coo = data_cont.tocoo()
    mask = coo.data >= margin_threshold

    valid_rows = coo.row[mask]
    valid_cols = coo.col[mask]
    n11 = coo.data[mask]

    data_df = pd.DataFrame({
        "product_name": product_names[valid_rows],
        "ae_name": ae_names[valid_cols],
        "events": n11,
        "product_aes": row_sums[valid_rows],
        "count_across_brands": col_sums[valid_cols],
    })

    # 5. Populate the VIGIPY Container
    DC = Container()
    DC.contingency = data_cont
    DC.product_index = product_names
    DC.ae_index = ae_names
    DC.data = data_df
    DC.N = float(data_df["events"].sum())
    DC.type = "contingency"
    return DC


def compute_contingency(data_frame, product_label, count_label, ae_label, margin_threshold):
    """Compute the contingency table as a filtered CSR sparse matrix."""
    df_clean = data_frame.dropna(subset=[product_label, ae_label])
    grouped = df_clean.groupby([product_label, ae_label], observed=True)[count_label].sum().reset_index()

    prod_cat = grouped[product_label].astype("category")
    ae_cat = grouped[ae_label].astype("category")

    mat = csr_matrix(
        (
            grouped[count_label].to_numpy(dtype=np.float64),
            (prod_cat.cat.codes.to_numpy(), ae_cat.cat.codes.to_numpy()),
        ),
        shape=(len(prod_cat.cat.categories), len(ae_cat.cat.categories)),
        dtype=np.float64,
    )

    # Simultaneous filtering of rows and columns based on margin_threshold
    row_mask = np.asarray(mat.sum(axis=1)).ravel() >= margin_threshold
    col_mask = np.asarray(mat.sum(axis=0)).ravel() >= margin_threshold

    return mat[row_mask, :][:, col_mask]


def convert_binary(
    data, product_label="name", ae_label="AE", use_counts=False, count_label="count", expand_counts=True
):
    """Convert product-event pairs into binary or count-weighted sparse matrices."""
    DC = Container()

    keep_cols = [product_label, ae_label]
    if count_label in data.columns:
        keep_cols.append(count_label)
    data = _sanitize_data(data, keep_cols)

    if count_label not in data.columns:
        data[count_label] = 1

    # Drop missing values to avoid invalid categorical codes
    subset_cols = ([product_label] if isinstance(product_label, str) else list(product_label)) + [ae_label]
    data = data.dropna(subset=subset_cols).copy()

    # Vectorized row expansion when expand_counts is enabled
    if not use_counts and expand_counts and data[count_label].max() > 1:
        data = data.loc[data.index.repeat(data[count_label].astype(int))].reset_index(drop=True)
        data[count_label] = 1

    if use_counts:
        group_list = ([product_label] if isinstance(product_label, str) else list(product_label)) + [ae_label]
        data = data.groupby(group_list, observed=True)[count_label].sum().reset_index()

        # Sparse construction of count-weighted events (replaces __transform_dataframe)
        y_sparse, ae_names = _build_sparse_indicator(
            data, ae_label, weights=data[count_label].to_numpy(), dtype=np.float64
        )
        # Sparse construction of binary products (replaces pd.get_dummies)
        X_sparse, product_names = _build_sparse_indicator(data, product_label, dtype=np.int8)

        DC.type = "binary_count"
    else:
        X_sparse, product_names = _build_sparse_indicator(data, product_label, dtype=np.int8)
        y_sparse, ae_names = _build_sparse_indicator(data, ae_label, dtype=np.int8)
        DC.type = "binary"

    # Expose as Sparse DataFrames (keeps existing VIGIPY API compatibility while saving ~95% RAM)
    DC.product_features = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=product_names)
    DC.event_outcomes = pd.DataFrame.sparse.from_spmatrix(y_sparse, columns=ae_names)
    DC.N = data.shape[0]
    DC.data = data

    return DC


def convert_multi_item(df, product_label=["name"], ae_label="AE", count_label="count", min_threshold=3):
    """Convert multi-product data into a VIGIPY Container with a sparse contingency table."""
    ae_counts = defaultdict(int)
    product_counts = defaultdict(int)

    for col in product_label:
        valid = df[df[col].notna() & (df[col] != "")]
        for ae, product, count_val in valid[[ae_label, col, count_label]].itertuples(index=False):
            ae_counts[ae] += count_val
            product_counts[product] += count_val

    df = _sanitize_data(df, [product_label, ae_label, count_label])

    result = []
    other_cols = [c for c in df.columns if c not in product_label]

    for row in df.itertuples(index=False):
        row_dict = dict(zip(df.columns, row))
        names = sorted({row_dict[x] for x in product_label if pd.notna(row_dict[x]) and row_dict[x] != ""})
        combos = chain.from_iterable(combinations(names, r) for r in range(1, len(names) + 1))

        base_data = {idx: row_dict[idx] for idx in other_cols}
        ae_val = row_dict[ae_label]

        for combo in combos:
            new_data = base_data.copy()
            new_data["product_name"] = "|".join(combo)
            new_data["product_aes"] = sum(product_counts[p] for p in combo)
            new_data["count_across_brands"] = ae_counts[ae_val]
            result.append(new_data)

    new_df = pd.DataFrame(result)
    new_df["events"] = new_df.groupby([ae_label, "product_name"])[count_label].transform("sum")
    new_df.rename(columns={ae_label: "ae_name"}, inplace=True)

    DC = Container()
    DC.contingency = compute_contingency(new_df, "product_name", count_label, "ae_name", min_threshold)
    DC.data = new_df[["ae_name", "product_name", "count_across_brands", "product_aes", "events"]].drop_duplicates()
    DC.N = new_df["events"].sum()
    DC.type = "contingency"

    return DC


def count(data, rows, cols):
    """Convert an input contingency table (CSR sparse matrix or DataFrame) to a flattened table."""
    if isinstance(data, csr_matrix):
        coo = data.tocoo()
        row_arr = np.asarray(rows)
        col_arr = np.asarray(cols)

        df = pd.DataFrame({
            "events": coo.data,
            "product_aes": row_arr[coo.row],
            "count_across_brands": col_arr[coo.col],
            "ae_name": coo.col,
            "product_name": coo.row,
        })
        return df[["events", "product_aes", "count_across_brands", "ae_name", "product_name"]]

    # Fallback if a dense DataFrame is passed
    stacked = data.stack()
    stacked = stacked[stacked > 0]
    df = stacked.reset_index()
    df.columns = ["product_name", "ae_name", "events"]
