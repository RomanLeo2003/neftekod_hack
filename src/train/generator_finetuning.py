import pandas as pd
from drugex.training.scorers.properties import Property, Uniqueness, AtomCounter
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import WeightedSum
from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.scorers.properties import Uniqueness
import os
from drugex.data.datasets import GraphFragDataSet
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.training.monitors import FileMonitor
from rdkit import Chem

from utils import is_valid_molecule

df = pd.read_csv('../../data/csvs/good_smiles.csv')
result_smiles = df['SMILES'].to_list()


# 🔬 1. Модель для предсказания времени индукционного окисления (например, RandomForestRegressor)
scorer_tio = QSPRPredScorer(
    model = SklearnModel(
        name="A2AR_ForestRegressor_hack_data_extended_with_submits",
        base_dir='data/models/qsar/',
    )
)

scorer_tio.setModifier(ClippedScore(lower_x=200, upper_x=600))  # время в минутах
scorers = [scorer_tio]
uniq_scorer = Uniqueness(modifier=ClippedScore(lower_x=0.0, upper_x=0.5))
# scorers.append(uniq_scorer)
oxygen_scorer = AtomCounter('O', modifier=ClippedScore(lower_x=17, upper_x=40))
# scorers.append(oxygen_scorer)

sascore = Property("SA", modifier=ClippedScore(lower_x=5, upper_x=3))
scorers.append(sascore)
thresholds.append(4)

qed = Property("QED", modifier=ClippedScore(lower_x=0.4, upper_x=1.0))
scorers.append(qed)
thresholds.append(0.8)


logp = Property("logP", modifier=ClippedScore(lower_x=1.0, upper_x=6.0))
scorers.append(logp)
thresholds.append(3)

mw = Property("MW", modifier=ClippedScore(lower_x=100, upper_x=950))
scorers.append(mw)
thresholds.append(500)

environment = DrugExEnvironment(
    scorers=scorers,
    reward_scheme=WeightedSum()  # можно заменить на ParetoCrowdingDistance, если нужно
)


fragmenter = dummyMolsFromFragments()
splitter = None

encoder = FragmentCorpusEncoder(
    fragmenter=Fragmenter(4, 4, 'brics'), 
    encoder=GraphFragmentEncoder(
        VocGraph(n_frags=4) 
    ),
    pairs_splitter=splitter, 
    n_proc=10,
    chunk_size=1
)

graph_input_folder = "data/encoded/graph"
if not os.path.exists(graph_input_folder):
    os.makedirs(graph_input_folder)
    
dataset = GraphFragDataSet(f"{graph_input_folder}/scaffolds.tsv", rewrite=True)

encoder.apply(list(result_smiles), encodingCollectors=[dataset])

GPUS = [0]


vocabulary = VocGraph.fromFile('../../models/pretrained_graph/Papyrus05.5_graph_trans_PT.vocab')
agent = GraphTransformer(voc_trg=vocabulary, use_gpus=GPUS)
agent.loadStatesFromFile('../../models/pretrained_graph/Papyrus05.5_graph_trans_PT.pkg')
prior = GraphTransformer(voc_trg=vocabulary, use_gpus=GPUS)
prior.loadStatesFromFile('../../models/pretrained_graph/Papyrus05.5_graph_trans_PT.pkg')

explorer = FragGraphExplorer(agent=agent, env=environment, mutate=prior, epsilon=0.1, use_gpus=GPUS)



data_path = f'{graph_input_folder}/scaffolds.tsv'
train_loader = GraphFragDataSet(data_path).asDataLoader(batch_size=1024, n_samples=100)
test_loader = GraphFragDataSet(data_path).asDataLoader(batch_size=1024, n_samples=100, n_samples_ratio=0.2)

monitor = FileMonitor("../../models/finetuned_graph_with_o_new/scaffolds.pkg", save_smiles=True) 
explorer.fit(train_loader, test_loader, monitor=monitor, epochs=5)

reinforced = GraphTransformer(voc_trg=VocGraph(), use_gpus=GPUS)
reinforced.loadStatesFromFile('../../models/finetuned_graph_with_o_new/scaffolds.pkg')
denovo_new = reinforced.generate(input_frags=result_smiles, num_samples=400, evaluator=environment)
denovo_filtered_new = denovo_new[denovo_new['SMILES'].apply(is_valid_molecule)].drop_duplicates(subset='SMILES').copy()

denovo_filtered_new['O_count'] = denovo_filtered_new['SMILES'].str.count('O')
submission_list = denovo_filtered_new.sort_values(by='O_count')['SMILES'].to_list()[-10:]
pd.DataFrame({'SMILES': submission_list}).to_csv('submission.csv', index=False, encoding='utf-8')