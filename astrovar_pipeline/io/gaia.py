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

    ids_to_download = []
    for source_id in source_ids:
        path = outdir / f"{source_id}.csv"
        if path.exists():
            continue  # already downloaded

        ids_to_download.append(source_id)

    batch_size = 10

    print(f"Downloading {len(ids_to_download)} lightcurves.")
    for i in tqdm.tqdm(
        range(0, len(ids_to_download), batch_size), "Downloading Gaia light curves"
    ):
        start, end = i, min(i + batch_size, len(ids_to_download))
        curr_ids = ids_to_download[start:end]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=u.UnitsWarning)

                lcs = Gaia.load_data(
                    curr_ids,
                    data_release="Gaia DR3",
                    data_structure="INDIVIDUAL",
                    retrieval_type="EPOCH_PHOTOMETRY",
                    valid_data=True,
                    format="fits",
                )

                for lc in lcs:
                    id = lc.split(" ")[2].strip(".fits")
                    path = f"{outdir}/{id}.csv"
                    table = lcs[lc][0]
                    table.write(path, overwrite=True)
                    downloaded.append((source_id, path))
        except Exception as e:
            print(f"Failed download for {curr_ids}: {e}")

    return downloaded
