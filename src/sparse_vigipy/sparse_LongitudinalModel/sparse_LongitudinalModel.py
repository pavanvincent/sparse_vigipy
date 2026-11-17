import pandas as pd
import numpy as np
from ..sparse-utils.sparse_data_prep import convert, convert_binary, convert_multi_item


class LongitudinalModel:

    CONVERSION_TYPES = {"base", "binary", "multi-item"}

    def __init__(self, dataframe, time_unit):
        self.time_unit = time_unit
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        # Sorting is mandatory and essential for .searchsorted() binary search
        self.data = dataframe.sort_values("date").reset_index(drop=True)
        self.date_groups = self.data.resample(time_unit, on="date")
        self.results = []

    def _convert(self, data, conversion_type, conversion_kwargs):
        if conversion_type not in self.CONVERSION_TYPES:
            raise ValueError(f"Provided `conversion_type` not in {self.CONVERSION_TYPES}")

        conversion_kwargs = conversion_kwargs or {}
        if conversion_type == "base":
            return convert(data, **conversion_kwargs)
        elif conversion_type == "binary":
            return convert_binary(data, **conversion_kwargs)
        elif conversion_type == "multi-item":
            return convert_multi_item(data, **conversion_kwargs)

    def _execute_disjoint_task(
        self,
        timestamp,
        subset_or_idx,
        model,
        include_gaps,
        conversion_type,
        conversion_kwargs,
        **kwargs,
    ):
        # Handle either a cumulative end index (int) or a disjoint DataFrame slice
        if isinstance(subset_or_idx, (int, float, np.integer)):
            if subset_or_idx == 0:
                return (timestamp, None) if include_gaps else None
            subset = self.data.iloc[: int(subset_or_idx)]
        else:
            if subset_or_idx is None or subset_or_idx.empty:
                return (timestamp, None) if include_gaps else None
            subset = subset_or_idx

        try:
            sub_container = self._convert(subset, conversion_type, conversion_kwargs)
            # Skip empty sparse contingency matrices when thresholds filter out all entries
            if hasattr(sub_container, "contingency") and sub_container.contingency.nnz == 0:
                return (timestamp, None) if include_gaps else None
            da_results = model(sub_container, **kwargs)
            return (timestamp, da_results)

        except (ValueError, IndexError, ZeroDivisionError):
            return (timestamp, None) if include_gaps else None

    def run_disjoint(
        self,
        model,
        include_gaps=True,
        conversion_type="base",
        conversion_kwargs=None,
        start_date=None,
        end_date=None,
        **kwargs,
    ):
        """Sequential disjoint version: evaluates independent time bins over [start_date, end_date]."""
        if conversion_kwargs is None:
            conversion_kwargs = {}

        # Extract subsets per resampled date group
        group_data = [
            (timestamp, self.date_groups.get_group(timestamp))
            for timestamp in self.date_groups.groups.keys()
        ]

        # Filter by start_date and end_date
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            group_data = [(ts, sub) for ts, sub in group_data if ts >= start_ts]

        if end_date is not None:
            end_ts = pd.to_datetime(end_date)
            group_data = [(ts, sub) for ts, sub in group_data if ts <= end_ts]

        raw_results = []
        for timestamp, subset in group_data:
            res = self._execute_disjoint_task(
                timestamp,
                subset,
                model,
                include_gaps,
                conversion_type,
                conversion_kwargs,
                **kwargs,
            )
            raw_results.append(res)

        self.results = [res for res in raw_results if res is not None]
        return self.results

    def run(
        self,
        model,
        include_gaps=True,
        conversion_type="base",
        conversion_kwargs=None,
        start_date=None,
        end_date=None,
        **kwargs,
    ):
        """Sequential cumulative version: evaluates expanding windows over [start_date, end_date]."""
        timestamps = list(self.date_groups.groups.keys())

        # Filter by start_date and end_date
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            timestamps = [ts for ts in timestamps if ts >= start_ts]

        if end_date is not None:
            end_ts = pd.to_datetime(end_date)
            timestamps = [ts for ts in timestamps if ts <= end_ts]

        # Binary search for cumulative right-hand boundaries
        end_indices = self.data["date"].searchsorted(timestamps, side="right")

        raw_results = []
        for timestamp, end_idx in zip(timestamps, end_indices):
            res = self._execute_disjoint_task(
                timestamp,
                end_idx,
                model,
                include_gaps,
                conversion_type,
                conversion_kwargs,
                **kwargs,
            )
            raw_results.append(res)

        self.results = [res for res in raw_results if res is not None]
        return self.results

    def regroup_dates(self, time_unit):
        self.time_unit = time_unit
        self.date_groups = self.data.resample(time_unit, on="date")
