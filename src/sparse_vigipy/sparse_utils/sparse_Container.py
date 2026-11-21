import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse, csr_matrix


def _extract_active_substances(value):
    """Extract and normalize active substances strictly matching fast_profiles.py."""
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
    def _build_from_clean_pairs(
        cls,
        pop_ids: pd.Index,
        prod_df: pd.DataFrame,   # columns: ["id", "prod"]
        reac_df: pd.DataFrame,   # columns: ["id", "reac"]
        mode: str = "individual",
    ) -> "Container":
        """Shared internal sparse builder for 'individual' (unique case) and 'contingency' (co-occurrence) modes."""
        if mode not in {"individual", "contingency"}:
            raise ValueError("mode must be 'individual' or 'contingency'")

        pop_set = set(pop_ids)
        prod_df = prod_df[prod_df["id"].isin(pop_set)].drop_duplicates(["id", "prod"])
        reac_df = reac_df[reac_df["id"].isin(pop_set)].drop_duplicates(["id", "reac"])

        if prod_df.empty or reac_df.empty:
            raise ValueError("No valid data remaining after filtering on population IDs.")

        s_rows = pop_ids.get_indexer(prod_df["id"])
        s_codes, product_names = pd.factorize(prod_df["prod"], sort=True)

        r_rows = pop_ids.get_indexer(reac_df["id"])
        r_codes, ae_names = pd.factorize(reac_df["reac"], sort=True)

        N_cases = len(pop_ids)
        sf = sparse.csr_matrix(
            (np.ones(len(s_rows), dtype=np.float64), (s_rows, s_codes)),
            shape=(N_cases, len(product_names)),
        )
        so = sparse.csr_matrix(
            (np.ones(len(r_rows), dtype=np.float64), (r_rows, r_codes)),
            shape=(N_cases, len(ae_names)),
        )

        # Matrix multiplication: counts exact number of cases sharing each (product, reaction) pair
        contingency = (sf.T @ so).tocoo()
        contingency.eliminate_zeros()

        prod_arr = product_names.to_numpy()
        ae_arr = ae_names.to_numpy()

        container = cls(params=False)
        container.contingency = contingency
        container.product_names = prod_arr
        container.ae_names = ae_arr
        container.product_index = prod_arr
        container.ae_index = ae_arr

        if mode == "individual":
            # UNIQUE CASE view: ror/prr/bcpnn/gps read sf and so for report-level margins and N = N_cases
            container.type = "individual"
            container.product_features = sf.astype(np.float32)
            container.event_outcomes = so.astype(np.float32)
            container.N = int(N_cases)
        else:
            # CO-OCCURRENCE view: omit product_features/event_outcomes -> margins = sums of contingency
            container.type = "contingency"
            container.N = float(contingency.data.sum())

        return container

    # =========================================================================
    # GENERIC METHOD (EUDRAVIGILANCE / RAW PARQUET TABLES)
    # =========================================================================

    @classmethod
    def from_raw_tables(
        cls,
        icsr_df: pd.DataFrame,
        suspects_df: pd.DataFrame,
        reactions_df: pd.DataFrame,
    ) -> "Container":
        """Build a sparse Container 100% aligned with fast_profiles.py and signal_methods.py."""
        pop_ids = pd.Index(icsr_df["safety_report_id"].dropna().unique())
        N = len(pop_ids)
        pop_set = set(pop_ids)

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

        contingency = (sf.astype(np.float64).T @ so.astype(np.float64)).tocoo()
        contingency.eliminate_zeros()

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

    # =========================================================================
    # FAERS METHODS (DEMO, DRUG, REAC)
    # =========================================================================

    @staticmethod
    def _prepare_faers_tables(
        demo_df: pd.DataFrame,
        drug_df: pd.DataFrame,
        reac_df: pd.DataFrame,
        deleted_ids: set | list | None = None,
        roles: tuple[str, ...] = ("PS", "SS"),
        split_combinations: bool = True,
    ) -> tuple[pd.Index, pd.DataFrame, pd.DataFrame]:
        # 1. Deduplicate at the patient level (caseid) -> keep latest primaryid
        demo = demo_df.sort_values("primaryid").drop_duplicates(subset=["caseid"], keep="last")
        if deleted_ids is not None:
            demo = demo[~demo["caseid"].isin(set(deleted_ids))]
        pop_ids = pd.Index(demo["primaryid"].dropna().unique())

        # 2. Filter and normalize suspect active substances
        drugs = drug_df[drug_df["role_cod"].isin(roles)].copy()
        raw_sub = drugs["prod_ai"].fillna(drugs["drugname"]).astype("string")
        if split_combinations:
            raw_sub = raw_sub.str.replace(r"[/;]", ",", regex=True)

        rows = []
        invalid = {"", "N/A", "NOT AVAILABLE", "NOT REPORTED", "UNKNOWN", "NAN", "NONE"}
        for pid, val in zip(drugs["primaryid"], raw_sub):
            if pd.isna(val):
                continue
            for item in str(val).split(","):
                clean = " ".join(item.strip().upper().split())
                if clean and clean not in invalid:
                    rows.append((pid, clean))
        prod_df = pd.DataFrame(rows, columns=["id", "prod"])

        # 3. Extract MedDRA Preferred Terms (PT)
        pts = reac_df["pt"].astype("string").str.strip()
        ok = pts.notna() & (pts != "")
        reac_clean = pd.DataFrame({
            "id": reac_df.loc[ok, "primaryid"].to_numpy(),
            "reac": pts.loc[ok].to_numpy(),
        })
        return pop_ids, prod_df, reac_clean

    @classmethod
    def from_faers_unique_case(
        cls,
        demo_df: pd.DataFrame,
        drug_df: pd.DataFrame,
        reac_df: pd.DataFrame,
        deleted_ids: set | list | None = None,
        roles: tuple[str, ...] = ("PS", "SS"),
        split_combinations: bool = True,
    ) -> "Container":
        """Build a FAERS Container in UNIQUE CASE mode (1 patient = 1 case)."""
        pop_ids, prod_df, reac_clean = cls._prepare_faers_tables(
            demo_df, drug_df, reac_df, deleted_ids, roles, split_combinations
        )
        return cls._build_from_clean_pairs(pop_ids, prod_df, reac_clean, mode="individual")

    @classmethod
    def from_faers_cooccurrence(
        cls,
        demo_df: pd.DataFrame,
        drug_df: pd.DataFrame,
        reac_df: pd.DataFrame,
        deleted_ids: set | list | None = None,
        roles: tuple[str, ...] = ("PS", "SS"),
        split_combinations: bool = True,
    ) -> "Container":
        """Build a FAERS Container in CARTESIAN CO-OCCURRENCE mode (every product x event pair counts)."""
        pop_ids, prod_df, reac_clean = cls._prepare_faers_tables(
            demo_df, drug_df, reac_df, deleted_ids, roles, split_combinations
        )
        return cls._build_from_clean_pairs(pop_ids, prod_df, reac_clean, mode="contingency")

    # =========================================================================
    # VAERS METHODS (VAERSDATA, VAERSVAX, VAERSSYMPTOMS)
    # =========================================================================

    @staticmethod
    def _prepare_vaers_tables(
        data_df: pd.DataFrame,
        vax_df: pd.DataFrame,
        symptoms_df: pd.DataFrame,
        vax_col: str = "VAX_NAME",
    ) -> tuple[pd.Index, pd.DataFrame, pd.DataFrame]:
        # 1. Unique case population by VAERS_ID
        pop_ids = pd.Index(data_df["VAERS_ID"].dropna().unique())

        # 2. Administered vaccines
        vax = vax_df[vax_df[vax_col].notna()].copy()
        vax_clean = vax[vax_col].astype("string").str.strip().str.upper()
        ok_v = (vax_clean != "") & (~vax_clean.isin({"UNKNOWN", "UNK", "N/A"}))
        prod_df = pd.DataFrame({
            "id": vax.loc[ok_v, "VAERS_ID"].to_numpy(),
            "prod": vax_clean.loc[ok_v].to_numpy(),
        })

        # 3. Symptoms (reshape wide SYMPTOM1..SYMPTOM5 columns into long format)
        symp_cols = [c for c in ("SYMPTOM1", "SYMPTOM2", "SYMPTOM3", "SYMPTOM4", "SYMPTOM5") if c in symptoms_df.columns]
        melted = symptoms_df.melt(
            id_vars=["VAERS_ID"],
            value_vars=symp_cols,
            value_name="reac",
        )[["VAERS_ID", "reac"]].dropna(subset=["reac"])

        melted["reac"] = melted["reac"].astype("string").str.strip()
        melted = melted[melted["reac"] != ""].rename(columns={"VAERS_ID": "id"})

        return pop_ids, prod_df, melted

    @classmethod
    def from_vaers_unique_case(
        cls,
        data_df: pd.DataFrame,
        vax_df: pd.DataFrame,
        symptoms_df: pd.DataFrame,
        vax_col: str = "VAX_NAME",
    ) -> "Container":
        """Build a VAERS Container in UNIQUE CASE mode (1 VAERS_ID = 1 case)."""
        pop_ids, prod_df, reac_clean = cls._prepare_vaers_tables(
            data_df, vax_df, symptoms_df, vax_col=vax_col
        )
        return cls._build_from_clean_pairs(pop_ids, prod_df, reac_clean, mode="individual")

    @classmethod
    def from_vaers_cooccurrence(
        cls,
        data_df: pd.DataFrame,
        vax_df: pd.DataFrame,
        symptoms_df: pd.DataFrame,
        vax_col: str = "VAX_NAME",
    ) -> "Container":
        """Build a VAERS Container in CARTESIAN CO-OCCURRENCE mode (every vaccine x symptom pair counts)."""
        pop_ids, prod_df, reac_clean = cls._prepare_vaers_tables(
            data_df, vax_df, symptoms_df, vax_col=vax_col
        )
        return cls._build_from_clean_pairs(pop_ids, prod_df, reac_clean, mode="contingency")

    # =========================================================================
    # SPARSE MATRIX ACCESSORS & EXPORT
    # =========================================================================

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


