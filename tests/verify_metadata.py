import sys
import os
from pathlib import Path
import numpy as np

def verify_all_environments():
    output_lmdb = "tests/sample.aselmdb"
    
    # 1. Check with Fairchem if available
    try:
        from fairchem.core.datasets import AseDBDataset
        print("\n--- Verifying with Fairchem AseDBDataset ---")
        dataset = AseDBDataset({"src": str(output_lmdb)})
        atoms = dataset.get_atoms(0)
        print(f"Chemical formula: {atoms.get_chemical_formula()}")
        for key in ['energy', 'string_val', 'list_val', 'dict_val']:
            val = atoms.info.get(key)
            print(f"  {key}: {val} (type: {type(val)})")
    except ImportError:
        print("\nFairchem not found in this environment.")

    # 2. Check with MACE if available
    try:
        from mace.tools.fairchem_dataset import AseDBDataset as MaceDataset
        print("\n--- Verifying with MACE AseDBDataset ---")
        dataset = MaceDataset({"src": str(output_lmdb)})
        atoms = dataset.get_atoms(0)
        print(f"Chemical formula: {atoms.get_chemical_formula()}")
        for key in ['energy', 'string_val', 'list_val', 'dict_val']:
            val = atoms.info.get(key)
            print(f"  {key}: {val} (type: {type(val)})")
    except ImportError:
        print("\nMACE not found in this environment.")

if __name__ == "__main__":
    verify_all_environments()
