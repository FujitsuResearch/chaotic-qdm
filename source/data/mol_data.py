import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pickle
import os
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset

class QDrugDataset(Dataset):
    """A PyTorch Dataset for processing molecular data from QM9 dataset.

    This class handles loading, processing, and normalizing molecular data
    for quantum chemistry tasks, supporting caching for efficiency.
    """

    def __init__(self, dat_name, n_qubits, load_from_cache: bool = False, file_path: str = './data/mol/', n_atoms: int = -1):
        """Initialize the dataset.

        Args:
            dat_name: Dataset name
            n_qubits: number of qubits.
            n_atoms (int): Specific number of atoms to filter molecules, -1 for no limit.
            load_from_cache (bool): Whether to load data from a cached file.
            file_path (str): Directory path for data and cache files.
        """
        self.dat_name = dat_name
        self.n_qubits = n_qubits
        self.file_path = file_path
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        self.atom_dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
        self.atomic_num_to_type = {6: 0, 7: 1, 8: 2, 9: 3}
        self.atom_types = 4 if dat_name == 'qm9' else None
        self.n_atoms = n_atoms  # If -1, no specific limit on number of atoms

        # Load or process raw molecular data
        self.raw_data = self._load_data(load_from_cache)
        
        if not self.raw_data:
            raise ValueError("No valid molecular data found. Check the SDF file or processing logic.")
        # Process the raw data into a dataset
        self.dataset, self.info, self.diff_minmax, self.min_val = self._process_data()

    def _load_data(self, load_from_cache: bool) -> List[Dict[str, Union[np.ndarray, str]]]:
        """Load raw molecular data from cache or process from SDF file.

        Args:
            load_from_cache (bool): Whether to load from cached pickle file.

        Returns:
            List[Dict[str, Union[np.ndarray, str]]]: List of molecular data dictionaries.
        """
        cache_file = os.path.join(self.file_path, f'raw_information_atoms_{self.n_atoms}.pkl')
        
        if load_from_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        if self.dat_name != 'qm9':
            raise ValueError(f"Unsupported dataset: {self.dat_name}")

        molecular_information = []
        mols = Chem.SDMolSupplier(os.path.join(self.file_path, 'gdb9.sdf'), removeHs=True, sanitize=True)
        
        for i, mol in enumerate(mols):
            try:
                n_atoms = mol.GetNumAtoms()
                if n_atoms > 9:
                    continue
                if self.n_atoms > 0 and n_atoms != self.n_atoms:
                    continue
                # Extract 3D positions
                pos = mols.GetItemText(i).split('\n')[4:4 + n_atoms]
                pos = np.array([[float(x) for x in line.split()[:3]] for line in pos])
                pos = self._determine_position(pos)
                
                if np.isnan(pos).any():
                    continue

                atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] 
                                    for atom in mol.GetAtoms()]).reshape(-1, 1)
                molecular_information.append({
                    'position': pos,
                    'atom_type': atom_type,
                    'smi': Chem.MolToSmiles(mol)
                })
            except Exception as e:
                print(f"Error processing molecule {i}: {str(e)}")
                continue

        # Cache processed data
        with open(cache_file, 'wb') as f:
            pickle.dump(molecular_information, f)
        
        return molecular_information

    def _determine_position(self, points: np.ndarray) -> np.ndarray:
        """Center and rotate the point cloud to a standard orientation.

        Args:
            points (np.ndarray): Array of 3D coordinates for atoms.

        Returns:
            np.ndarray: Transformed coordinates.
        """
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        translated_points = points - centroid

        # Rotate to align first point
        first_point = translated_points[0]
        x, y, z = first_point

        # Compute rotation angles
        norm_xz = np.sqrt(x ** 2 + z ** 2)
        sin_angle_1 = -x / norm_xz if norm_xz != 0 else 0
        cos_angle_1 = z / norm_xz if norm_xz != 0 else 1

        norm_xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        sin_angle_2 = y / norm_xyz if norm_xyz != 0 else 0
        cos_angle_2 = np.sqrt(x ** 2 + z ** 2) / norm_xyz if norm_xyz != 0 else 1

        # Define rotation matrix
        rotation_matrix = np.array([
            [cos_angle_1, 0, sin_angle_1],
            [sin_angle_2 * sin_angle_1, cos_angle_2, -sin_angle_2 * cos_angle_1],
            [-cos_angle_2 * sin_angle_1, sin_angle_2, cos_angle_2 * cos_angle_1]
        ])

        return np.dot(rotation_matrix, translated_points.T).T

    def _process_data(self) -> Tuple[np.ndarray, List[Dict[str, Union[np.ndarray, str]]], float, np.ndarray]:
        """Process raw data into normalized dataset format.

        Returns:
            Tuple containing:
            - dataset (np.ndarray): Processed dataset.
            - info (List[Dict]): Metadata for each molecule.
            - diff_minmax (float): Normalization scale factor.
            - min_val (np.ndarray): Minimum position values for normalization.
        """
        positions = []
        for item in self.raw_data:
            if np.isnan(item['position']).any():
                print(f"NaN in position data for molecule, skipping: {item['smi']}")
                continue
            positions.append(item['position'])

        positions = np.concatenate(positions, axis=0)
        min_val = np.min(positions, axis=0)
        max_val = np.max(positions, axis=0)
        diff_minmax = np.max(max_val - min_val)

        dataset = np.zeros((len(self.raw_data), 2 ** self.n_qubits + 1))

        info = []
        max_len = 0

        for i, item in enumerate(self.raw_data):
            position = (item['position'] - min_val) / diff_minmax
            atom_type = np.eye(self.atom_types)[item['atom_type'].astype(int)].squeeze(1)
            
            # Compute auxiliary vector
            aux_vec = np.array([np.sqrt(3 - np.sum(position[j] ** 2)) for j in range(len(position))])
            
            # Combine position and atom type
            tmp = np.concatenate((position, atom_type), axis=1).flatten()
            #print(f"Processed item {i}: position shape {position.shape}, atom_type shape {atom_type.shape}, tmp shape {tmp.shape}, aux_vec shape {aux_vec.shape}")

            dataset[i, :len(tmp)] = tmp
            dataset[i, len(tmp):len(tmp) + len(position)] = aux_vec
            dataset[i] = dataset[i] / (2 * np.sqrt(len(position)))
            dataset[i, -1] = len(position)
            
            if len(position) > max_len:
                max_len = len(position)
            
            info.append({'x': dataset[i], 'smi': item['smi']})

        return dataset, info, diff_minmax, min_val

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        """Get dataset item by index.

        Args:
            index (int): Index of the item.

        Returns:
            Dict[str, Union[np.ndarray, str]]: Dictionary containing processed data and SMILES.
        """
        return self.info[index]

    def __len__(self) -> int:
        """Get the number of items in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)
    

def filter_dataset_by_properties(
    dataset: QDrugDataset,
    target_num_rings: int = None,
    target_n_atoms: int = None,
    min_mol_weight: float = None,
    max_mol_weight: float = None,
    save_path: str = None
) -> tuple:
    """Filter QDrugDataset by number of rings, number of atoms, and molecular weight.

    Args:
        dataset (QDrugDataset): Loaded QDrugDataset instance.
        target_num_rings (int, optional): Number of rings to filter by (e.g., 0, 1, 2).
        target_n_atoms (int, optional): Number of atoms to filter by (1 to 9 for QM9).
        min_mol_weight (float, optional): Minimum molecular weight (Da).
        max_mol_weight (float, optional): Maximum molecular weight (Da).
        save_path (str, optional): Path to save the filtered dataset (pickle file).

    Returns:
        tuple: (filtered_dataset, filtered_info)
            - filtered_dataset (np.ndarray): Filtered dataset array.
            - filtered_info (list): Filtered list of {'x': array, 'smi': str} dictionaries.

    Raises:
        ValueError: If no molecules match the criteria or inputs are invalid.
    """
    # Validate inputs
    if target_n_atoms is not None and not 1 <= target_n_atoms <= 9:
        target_n_atoms = None  # Ignore invalid values
        #raise ValueError(f"target_n_atoms must be between 1 and 9, got {target_n_atoms}")
    if target_num_rings is not None and target_num_rings < 0:
        target_num_rings = None  # Ignore negative values
        #raise ValueError(f"target_num_rings must be non-negative, got {target_num_rings}")

    # Initialize mask
    mask = np.ones(len(dataset), dtype=bool)

    # Filter by n_atoms
    if target_n_atoms is not None:
        n_atoms = dataset.dataset[:, -1].astype(int)
        mask = mask & (n_atoms == target_n_atoms)

    # Filter by molecular properties using RDKit
    filtered_indices = []
    for i in range(len(dataset.info)):
        if not mask[i]:
            continue
        mol = Chem.MolFromSmiles(dataset.info[i]['smi'])
        if mol is None:
            print(f"Skipping molecule {i}: Invalid SMILES {dataset.info[i]['smi']}")
            continue

        # Check number of rings
        if target_num_rings is not None:
            if rdMolDescriptors.CalcNumRings(mol) != target_num_rings:
                continue

        # Check molecular weight
        if min_mol_weight is not None or max_mol_weight is not None:
            mol_weight = Descriptors.MolWt(mol)
            if min_mol_weight is not None and mol_weight < min_mol_weight:
                continue
            if max_mol_weight is not None and mol_weight > max_mol_weight:
                continue

        filtered_indices.append(i)

    if not filtered_indices:
        raise ValueError(
            f"No molecules found with num_rings={target_num_rings or 'any'}, "
            f"n_atoms={target_n_atoms or 'any'}, "
            f"mol_weight=[{min_mol_weight or '-inf'}, {max_mol_weight or 'inf'}]"
        )

    # Filter dataset and info
    filtered_dataset = dataset.dataset[filtered_indices]
    filtered_info = [dataset.info[i] for i in filtered_indices]

    # Check norms of filtered feature vectors
    norms = np.linalg.norm(filtered_dataset[:, :-1], axis=1)
    print(f"Filtered {len(filtered_indices)} molecules")
    print(f"Norm statistics: mean={np.mean(norms):.6f}, std={np.std(norms):.6f}, "
                f"min={np.min(norms):.6f}, max={np.max(norms):.6f}")

    # Save filtered dataset (optional)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dataset': filtered_dataset,
                'info': filtered_info,
                'diff_minmax': dataset.diff_minmax,
                'min_val': dataset.min_val
            }, f)
        print(f"Saved filtered dataset to {save_path}")

    return filtered_dataset, filtered_info