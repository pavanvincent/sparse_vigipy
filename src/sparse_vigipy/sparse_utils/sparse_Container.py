import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse, csr_matrix


def _extract_active_substances(value):
    """Extraction et normalisation strictement identique à fast_profiles.py."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    low = text.casefold()
    if low == "not reported" or low.startswith("not available") or low.startswith("[truncated"):
        return []
    left = text.find("[")
    right = text.find("]", left + 1)
    if left != -1 and right != -1 and right > left + 1:
        substances = [x.strip() for x in text[left + 1:right].split(",") if x.strip()]
    else:
        base = text.split(" (", 1)[0].strip()
        substances = [x.strip() for x in base.split(",") if x.strip()] if base else []

    invalid = {"", "N/A", "NOT AVAILABLE", "NOT REPORTED", "UNKNOWN"}
    return [
        " ".join(name.upper().split())
        for name in substances
        if " ".join(name.upper().split()) not in invalid
    ]


class Container:
    def __init__(self, params=False):
        if params:
            self.param = dict()

    @classmethod
    def from_raw_tables(
        cls,
        icsr_df: pd.DataFrame,
        suspects_df: pd.DataFrame,
        reactions_df: pd.DataFrame,
    ) -> "Container":
        """Construit un Container creux 100 % aligné sur fast_profiles.py et signal_methods.py."""
        # 1. Population unique (N)
        pop_ids = pd.Index(icsr_df["safety_report_id"].dropna().unique())
        N = len(pop_ids)
        pop_set = set(pop_ids)

        # 2. Substances actives dédoublonnées par signalement
        sub_rows = []
        for row in suspects_df[["safety_report_id", "drug"]].itertuples(index=False):
            if row.safety_report_id in pop_set:
                for sub in _extract_active_substances(row.drug):
                    sub_rows.append((row.safety_report_id, sub))

        if sub_rows:
            sub_df = pd.DataFrame(sub_rows, columns=["id", "substance"]).drop_duplicates(["id", "substance"])
            s_rows = pop_ids.get_indexer(sub_df["id"])
            s_codes, product_names = pd.factorize(sub_df["substance"], sort=True)
            sf = sparse.csr_matrix(
                (np.ones(len(s_rows), dtype=np.float32), (s_rows, s_codes)),
                shape=(N, len(product_names)),
            )
        else:
            product_names = pd.Index([])
            sf = sparse.csr_matrix((N, 0), dtype=np.float32)

        # 3. Réactions dédoublonnées par signalement
        pts = reactions_df["reaction_pt"].astype("string").str.strip()
        valid_rx = pts.notna() & (pts != "") & reactions_df["safety_report_id"].isin(pop_set)

        rx_df = pd.DataFrame({
            "id": reactions_df.loc[valid_rx, "safety_report_id"].to_numpy(),
            "pt": pts.loc[valid_rx].to_numpy(),
        }).drop_duplicates(["id", "pt"])

        if not rx_df.empty:
            r_rows = pop_ids.get_indexer(rx_df["id"])
            r_codes, ae_names = pd.factorize(rx_df["pt"], sort=True)
            so = sparse.csr_matrix(
                (np.ones(len(r_rows), dtype=np.float32), (r_rows, r_codes)),
                shape=(N, len(ae_names)),
            )
        else:
            ae_names = pd.Index([])
            so = sparse.csr_matrix((N, 0), dtype=np.float32)

        # 4. Matrice de contingence creuse (Produits x Réactions)
        contingency = (sf.astype(np.float64).T @ so.astype(np.float64)).tocoo()
        contingency.eliminate_zeros()

        # 5. Assemblage du Container
        container = cls(params=False)
        container.type = "individual"
        container.product_features = sf
        container.event_outcomes = so
        container.contingency = contingency

        prod_arr = product_names.to_numpy()
        ae_arr = ae_names.to_numpy()
        container.product_names = prod_arr
        container.ae_names = ae_arr
        container.product_index = prod_arr
        container.ae_index = ae_arr
        container.N = N
        return container

    def get_sparse_features(self):
        if not hasattr(self, "product_features"):
            raise AttributeError("Container has no 'product_features' attribute.")
        if issparse(self.product_features):
            return self.product_features.tocsr()
        if isinstance(self.product_features, pd.DataFrame) and hasattr(self.product_features, "sparse"):
            return self.product_features.sparse.to_coo().tocsr()
        return csr_matrix(self.product_features)

    def get_sparse_outcomes(self):
        if not hasattr(self, "event_outcomes"):
            raise AttributeError("Container has no 'event_outcomes' attribute.")
        if issparse(self.event_outcomes):
            return self.event_outcomes.tocsr()
        if isinstance(self.event_outcomes, pd.DataFrame) and hasattr(self.event_outcomes, "sparse"):
            return self.event_outcomes.sparse.to_coo().tocsr()
        return csr_matrix(self.event_outcomes)

    def export(self, name, index=False):
        with pd.ExcelWriter(name) as writer:
            self.signals.to_excel(writer, sheet_name="Signals", index=index)
            self.all_signals.to_excel(writer, sheet_name="all_data", index=index)


