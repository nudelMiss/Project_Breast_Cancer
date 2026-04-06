#!/usr/bin/env python3
"""
Preprocess CORUM core complexes from GSDB R serialized file.

Extracts CORUM_core_complexes from the GSDB.list.of.lists object and saves
as a flat TSV with columns: complex_id, gene.

Usage:
    python scripts/preprocess_corum.py \
        --gsdb_path GSDB.list.of.lists.RData \
        --output_path resources/corum_core_complexes.tsv
"""
import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyreadr  # type: ignore
except ImportError:
    pyreadr = None

try:
    import rdata  # type: ignore
except ImportError:
    rdata = None


def load_gsdb(gsdb_path: Path) -> dict:
    """
    Load the GSDB.list.of.lists R serialized file.

    Args:
        gsdb_path: Path to the GSDB.list.of.lists file

    Returns:
        Dictionary with GSDB contents
    """
    print(f"Loading GSDB from: {gsdb_path}", flush=True)

    if pyreadr is not None:
        result = pyreadr.read_r(str(gsdb_path))
        print(f"Loaded {len(result)} objects from GSDB via pyreadr", flush=True)
        return result

    if rdata is None:
        raise ImportError(
            "Neither pyreadr nor rdata is installed. Install one of them to read GSDB files."
        )

    # Pure-Python fallback for environments where pyreadr compilation fails.
    path_str = str(gsdb_path)
    if path_str.endswith('.rds'):
        obj = rdata.read_rds(path_str)
        result = {'rds_object': obj}
    else:
        result = rdata.read_rda(path_str)
        if not isinstance(result, dict):
            result = {'rdata_object': result}

    print(f"Loaded {len(result)} objects from GSDB via rdata", flush=True)
    return result


def extract_corum_complexes(gsdb_data: dict) -> pd.DataFrame:
    """
    Extract CORUM_core_complexes and flatten to TSV format.

    The GSDB structure can be:
    - A single dataframe with all data
    - A dict/OrderedDict with named gene sets

    Args:
        gsdb_data: Loaded GSDB data from pyreadr

    Returns:
        DataFrame with columns: complex_id, complex_name, gene
    """
    # pyreadr returns an OrderedDict; the first/only key contains the data
    # We need to find CORUM_core_complexes within

    # Check what keys we have
    print(f"GSDB keys: {list(gsdb_data.keys())[:10]}...", flush=True)

    # The structure depends on how the R object was saved
    # Try to find CORUM_core_complexes
    corum_key = None
    for key in gsdb_data.keys():
        if 'CORUM_core_complexes' in key or key == 'CORUM_core_complexes':
            corum_key = key
            break

    if corum_key:
        # CORUM_core_complexes is a direct key
        corum_data = gsdb_data[corum_key]
        print(f"Found CORUM_core_complexes as direct key", flush=True)
    else:
        # Check if it's nested - the main object might be a list of lists
        # In this case, the first dataframe might contain all the data
        first_key = list(gsdb_data.keys())[0]
        main_data = gsdb_data[first_key]

        if isinstance(main_data, pd.DataFrame):
            print(f"Main data is a DataFrame with columns: {list(main_data.columns)}", flush=True)
            # Check if there's a column indicating the gene set source
            if 'CORUM_core_complexes' in main_data.values:
                corum_data = main_data[main_data.iloc[:, 0] == 'CORUM_core_complexes']
            else:
                # Use the whole dataframe and filter later
                corum_data = main_data
        elif isinstance(main_data, dict):
            if 'CORUM_core_complexes' in main_data:
                corum_data = main_data['CORUM_core_complexes']
            else:
                raise KeyError("Could not find CORUM_core_complexes in GSDB data")
        else:
            raise TypeError(f"Unexpected data type: {type(main_data)}")

    # Now process corum_data into flat format
    rows = []

    def _iter_genes(obj):
        """Yield gene symbols from heterogeneous container types."""
        if obj is None:
            return

        if isinstance(obj, (str, bytes)):
            text = obj.decode() if isinstance(obj, bytes) else obj
            text = text.strip()
            if text:
                yield text
            return

        if isinstance(obj, Mapping):
            for key in ('gene', 'genes', 'symbol', 'symbols', 'members'):
                if key in obj:
                    for gene in _iter_genes(obj[key]):
                        yield gene
                    return
            for value in obj.values():
                for gene in _iter_genes(value):
                    yield gene
            return

        if isinstance(obj, np.ndarray):
            for item in obj.ravel().tolist():
                for gene in _iter_genes(item):
                    yield gene
            return

        if isinstance(obj, pd.Series):
            for item in obj.tolist():
                for gene in _iter_genes(item):
                    yield gene
            return

        if isinstance(obj, Iterable):
            for item in obj:
                for gene in _iter_genes(item):
                    yield gene
            return

        # Scalar fallback
        text = str(obj).strip()
        if text:
            yield text

    if isinstance(corum_data, pd.DataFrame):
        # If it's already a dataframe, adapt based on structure
        print(f"CORUM data is DataFrame with shape {corum_data.shape}", flush=True)
        print(f"Columns: {list(corum_data.columns)}", flush=True)

        # Expected: each row is a gene, with complex_id and complex_name columns
        # Or: each row is a complex with a genes column containing a list
        if 'gene' in corum_data.columns:
            return corum_data[['complex_id', 'complex_name', 'gene']].copy()
        else:
            # Try to interpret the structure
            for idx, row in corum_data.iterrows():
                complex_id = str(idx)
                complex_name = str(row.get('name', idx))
                genes = row.get('genes', [])
                if isinstance(genes, str):
                    genes = genes.split(',')
                for gene in genes:
                    rows.append({
                        'complex_id': complex_id,
                        'complex_name': complex_name,
                        'gene': str(gene).strip()
                    })
    elif isinstance(corum_data, dict):
        # Dict mapping complex_name -> list of genes
        print(f"CORUM data is dict with {len(corum_data)} complexes", flush=True)
        for idx, (complex_name, genes) in enumerate(corum_data.items()):
            found_gene = False
            for gene in _iter_genes(genes):
                found_gene = True
                rows.append({
                    'complex_id': str(idx),
                    'complex_name': str(complex_name),
                    'gene': str(gene).strip(),
                })
            if not found_gene:
                continue
    elif isinstance(corum_data, list):
        # List of complexes
        print(f"CORUM data is list with {len(corum_data)} elements", flush=True)
        for idx, item in enumerate(corum_data):
            if isinstance(item, dict):
                complex_name = item.get('name', f'Complex_{idx}')
                genes = item.get('genes', [])
            elif isinstance(item, (list, tuple)):
                complex_name = f'Complex_{idx}'
                genes = item
            else:
                continue

            for gene in genes:
                rows.append({
                    'complex_id': str(idx),
                    'complex_name': str(complex_name),
                    'gene': str(gene).strip()
                })
    else:
        raise TypeError(f"Cannot process CORUM data of type: {type(corum_data)}")

    df = pd.DataFrame(rows, columns=['complex_id', 'complex_name', 'gene'])
    print(f"Extracted {len(df)} gene-complex pairs", flush=True)
    return df


def normalize_gene_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize gene names: strip whitespace, convert to uppercase.

    Args:
        df: DataFrame with 'gene' column

    Returns:
        DataFrame with normalized gene names
    """
    df = df.copy()
    df['gene'] = df['gene'].astype(str).str.strip().str.upper()
    # Remove empty genes
    df = df[df['gene'] != '']
    df = df[df['gene'] != 'NAN']
    df = df[df['gene'] != 'NONE']
    # Remove duplicates (same gene in same complex)
    df = df.drop_duplicates(subset=['complex_id', 'gene'])
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CORUM complexes from GSDB R file'
    )
    parser.add_argument(
        '--gsdb_path', type=str, default='GSDB.list.of.lists.RData',
        help='Path to GSDB.list.of.lists file'
    )
    parser.add_argument(
        '--output_path', type=str, default='resources/corum_core_complexes.tsv',
        help='Output TSV path'
    )
    args = parser.parse_args()

    gsdb_path = Path(args.gsdb_path)
    output_path = Path(args.output_path)

    if not gsdb_path.exists():
        # Backward compatible fallbacks for existing project files.
        fallback_candidates = [
            Path('GSDB.list.of.lists.RData'),
            Path('GSDB.list.of.lists.rds'),
            Path('GSDB.list.of.lists'),
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                gsdb_path = candidate
                print(f"Input not found, using fallback GSDB file: {gsdb_path}", flush=True)
                break
        else:
            raise FileNotFoundError(f"GSDB file not found: {gsdb_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and process
    gsdb_data = load_gsdb(gsdb_path)
    df = extract_corum_complexes(gsdb_data)
    df = normalize_gene_names(df)

    # Save exactly the requested flat output schema.
    out_df = df[['complex_id', 'gene']].copy()
    out_df.to_csv(output_path, sep='\t', index=False)

    # Summary stats
    n_complexes = out_df['complex_id'].nunique()
    n_genes = out_df['gene'].nunique()

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Total rows: {len(out_df)}", flush=True)
    print(f"Unique complexes: {n_complexes}", flush=True)
    print(f"Unique genes: {n_genes}", flush=True)
    print(f"Genes per complex (mean): {len(df) / n_complexes:.1f}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
