from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from astroquery.gaia import Gaia
import astropy.units as u
import warnings
import tqdm


def login(username: str | None = None, password: str | None = None):
    """Login to Gaia TAP server."""
    if username and password:
        Gaia.login(user=username, password=password)
    else:
        try:
            Gaia.login()  # use .netrc if available
        except Exception:
            pass


def read_source_ids(csv_path: str) -> List[int]:
    df = pd.read_csv(csv_path)
    if "source_id" not in df.columns:
        raise ValueError("CSV must contain a 'source_id' column.")
    return df["source_id"].astype("int64").tolist()


def fetch_gaia_epoch_photometry(
    source_ids: List[int], outdir: str
) -> List[Tuple[int, str]]:
    """
    Fetch Gaia DR3 epoch photometry for a list of source IDs using the
    official archive loader (`Gaia.load_data`), saving each light curve as CSV.

    Parameters
    ----------
    source_ids : list[int]
        Gaia DR3 source IDs to download.
    outdir : str
        Directory to save light-curve CSV files.

    Returns
    -------
    list of (source_id, filepath)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for source_id in tqdm.tqdm(source_ids, desc="Downloading Gaia light curves"):
        path = outdir / f"{source_id}.csv"
        if path.exists():
            continue  # skip already-downloaded

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=u.UnitsWarning)

                lc = Gaia.load_data(
                    [source_id],
                    data_release="Gaia DR3",
                    data_structure="INDIVIDUAL",
                    retrieval_type="EPOCH_PHOTOMETRY",
                    valid_data=True,
                    format="fits",
                )
            # Convert FITS to pandas DataFrame
            gaia_lc = lc[list(lc.keys())[0]][0]
            gband_frame = gaia_lc.to_pandas()
            gband_frame.to_csv(path, index=False)
            downloaded.append((source_id, str(path)))
        except Exception as e:
            print(f"❌ Failed for {source_id}: {e}")

    return downloaded
