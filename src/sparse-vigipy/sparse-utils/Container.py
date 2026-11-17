import pandas as pd
from scipy.sparse import issparse, csr_matrix


class Container:
    def __init__(self, params=False):
        if params:
            self.param = dict()

    def get_sparse_features(self):
        """Return product_features as a SciPy CSR sparse matrix."""
        if not hasattr(self, "product_features"):
            raise AttributeError("Container has no 'product_features' attribute.")
        if issparse(self.product_features):
            return self.product_features.tocsr()
        if isinstance(self.product_features, pd.DataFrame) and hasattr(self.product_features, "sparse"):
            return self.product_features.sparse.to_coo().tocsr()
        return csr_matrix(self.product_features)

    def get_sparse_outcomes(self):
        """Return event_outcomes as a SciPy CSR sparse matrix."""
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
