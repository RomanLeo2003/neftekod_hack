from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import pandas as pd
from drugex.training.scorers.smiles import SmilesChecker
import pandas as pd
# Создаем калькулятор с нужным дескриптором
from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=True)

allowed_atoms = {'C', 'H', 'O', 'N', 'P', 'S'}

# SMARTS-шаблоны для фенола и аминов
phenol_pattern = Chem.MolFromSmarts('c1ccc(cc1)O')  # Фенольная группа
amine_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')  # Алифатические амины (но не амиды)

def is_valid_molecule(smiles):
    """
    Проверяет молекулу на соответствие критериям:
    - Нейтральность, отсутствие радикалов
    - Масса <= 1000
    - Только разрешённые атомы (C, H, O, N, P, S)
    - logP > 1
    - Наличие фенольной ИЛИ аминной группы
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # 1. Проверка нейтральности и отсутствия радикалов
    if Chem.GetFormalCharge(mol) != 0:
        return False
    if any(atom.GetNumRadicalElectrons() != 0 for atom in mol.GetAtoms()):
        return False

    # 2. Молекулярная масса ≤ 1000
    if Descriptors.MolWt(mol) > 1000:
        return False

    # 3. Только разрешённые атомы
    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not atoms.issubset(allowed_atoms):
        return False

    # 4. logP > 1 (растворимость в гексане)
    if Crippen.MolLogP(mol) <= 1:
        return False

    if '+' in smiles:
        return False
    # 5. Наличие фенольной ИЛИ аминной группы
    has_phenol = mol.HasSubstructMatch(phenol_pattern)
    has_amine = mol.HasSubstructMatch(amine_pattern)
    if not (has_phenol or has_amine):
        return False

    if SmilesChecker.checkSmiles([smiles])['Valid'].to_list()[0] != 1.0:
        return False
    
    return True


# Функция для расчета AATSC0d
def compute_aatsc0d(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        result = calc(mol)
        return result['AATSC0d']
    except:
        return None
