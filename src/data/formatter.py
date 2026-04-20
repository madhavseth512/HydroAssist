"""Data Formatter — converts raw pumping test DataFrame to the standard schema."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.csv_inspector import InspectionResult

logger = logging.getLogger(__name__)

# ── Conversion constants ──────────────────────────────────────────────────────

TIME_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
DRAWDOWN_TO_METERS = {"meters": 1.0, "feet": 0.3048}


@dataclass
class FormattedData:
    df: pd.DataFrame               # Columns: time_s, drawdown_m[, well_id]
    original_shape: tuple
    final_shape: tuple
    conversions_applied: list = field(default_factory=list)
    rows_removed: list = field(default_factory=list)


class DataFormatter:
    """
    Converts a raw pumping test DataFrame + InspectionResult into a clean,
    standardised DataFrame ready for the calculator agents.

    Standard output schema:
        time_s      float   elapsed time in seconds
        drawdown_m  float   drawdown in meters
        well_id     str     (optional, only present for multi-well datasets)
    """

    def format(self, df: pd.DataFrame, result: InspectionResult) -> FormattedData:
        """
        Apply all formatting and unit conversions.

        Args:
            df:     Raw DataFrame from pd.read_csv
            result: InspectionResult from CSVInspector.inspect()

        Returns:
            FormattedData with clean DataFrame and change log
        """
        original_shape = df.shape
        conversions: list[str] = []
        removed: list[str] = []

        # 1. Extract relevant columns only
        cols_to_keep = [result.time_col] + result.drawdown_cols
        if result.well_id_col:
            cols_to_keep.append(result.well_id_col)
        out = df[cols_to_keep].copy()

        # 2. Rename to standard names
        rename_map = {result.time_col: "_time_raw"}
        if result.well_id_col:
            rename_map[result.well_id_col] = "well_id"
        # For single drawdown column, rename directly; for multi-well wide format, melt later
        if len(result.drawdown_cols) == 1:
            rename_map[result.drawdown_cols[0]] = "_dd_raw"
        out = out.rename(columns=rename_map)

        # 3. Handle multi-well wide format → long format
        if len(result.drawdown_cols) > 1:
            id_vars = ["_time_raw"] + (["well_id"] if result.well_id_col else [])
            out = out.melt(
                id_vars=id_vars,
                value_vars=result.drawdown_cols,
                var_name="well_id",
                value_name="_dd_raw",
            )
            conversions.append(f"Wide format ({len(result.drawdown_cols)} wells) melted to long format.")

        # 4. Coerce to numeric, drop unparseable rows
        before = len(out)
        out["_time_raw"] = pd.to_numeric(out["_time_raw"], errors="coerce")
        out["_dd_raw"]   = pd.to_numeric(out["_dd_raw"],   errors="coerce")
        out = out.dropna(subset=["_time_raw", "_dd_raw"])
        dropped = before - len(out)
        if dropped:
            removed.append(f"{dropped} row(s) dropped — non-numeric values in time or drawdown.")

        # 5. Convert time to seconds
        time_factor = TIME_TO_SECONDS.get(result.time_unit, 1.0)
        if result.time_unit != "seconds" and result.time_unit != "unknown":
            out["time_s"] = out["_time_raw"] * time_factor
            conversions.append(f"Time: {result.time_unit} → seconds (×{time_factor})")
        else:
            out["time_s"] = out["_time_raw"]
            if result.time_unit == "unknown":
                logger.warning("Time unit unknown — assuming seconds.")
                conversions.append("Time unit unknown — assumed seconds.")

        # 6. Convert drawdown to meters
        dd_factor = DRAWDOWN_TO_METERS.get(result.drawdown_unit, 1.0)
        if result.drawdown_unit == "feet":
            out["drawdown_m"] = out["_dd_raw"] * dd_factor
            conversions.append(f"Drawdown: feet → meters (×{dd_factor})")
        else:
            out["drawdown_m"] = out["_dd_raw"]
            if result.drawdown_unit == "unknown":
                logger.warning("Drawdown unit unknown — assuming meters.")
                conversions.append("Drawdown unit unknown — assumed meters.")

        # 7. Remove t=0 rows (log transform undefined)
        t0_mask = out["time_s"] == 0
        if t0_mask.any():
            out = out[~t0_mask]
            removed.append(f"{t0_mask.sum()} row(s) removed: t=0 (undefined under log transform).")

        # 8. Remove duplicate timestamps (keep first)
        group_cols = ["well_id", "time_s"] if "well_id" in out.columns else ["time_s"]
        dup_mask = out.duplicated(subset=group_cols, keep="first")
        if dup_mask.any():
            out = out[~dup_mask]
            removed.append(f"{dup_mask.sum()} duplicate timestamp(s) removed (kept first occurrence).")

        # 9. Sort by time ascending
        sort_cols = ["well_id", "time_s"] if "well_id" in out.columns else ["time_s"]
        out = out.sort_values(sort_cols).reset_index(drop=True)

        # 10. Keep only final schema columns
        final_cols = ["time_s", "drawdown_m"]
        if "well_id" in out.columns:
            final_cols.append("well_id")
        out = out[final_cols]

        logger.info(
            f"Formatted: {original_shape} → {out.shape}  |  "
            f"conversions={len(conversions)}  removed={len(removed)}"
        )

        return FormattedData(
            df=out,
            original_shape=original_shape,
            final_shape=out.shape,
            conversions_applied=conversions,
            rows_removed=removed,
        )
