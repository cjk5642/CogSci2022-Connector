# Fast and frugal memory search for communcation
# **Authors**: Collin J. Kovacs, Jasper M. Wilson, and Abhilasha A. Kumar
# 
# > *Communication involves searching for optimal utterances
# within memory and then evaluating those utterances against
# a target goal. However, whether this search process converges
# onto the optimal response relatively quickly, or involves more
# strategic decision-making to evaluate different candidates re-
# mains understudied. In this work, speakers generated single
# word “clues” that would enable a listener to correctly iden-
# tify a pair of words among several distractor words. Speak-
# ers and listeners generated candidates before producing final
# responses. Each player was biased towards the first candi-
# date(s) they generated, even when this candidate was sub-
# optimal compared to other candidates. Furthermore, straying
# away from the initial “patch” of responses decreased accuracy
# in the game. Overall, these findings suggest that individuals
# tend to identify the relevant semantic cluster early on during
# semantic search, and are likely to employ the “take-the-first”
# strategy in ambiguous semantic contexts.*

# Imports
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Load Data
def create_new_data_path(path:str = "new_data") -> str:
  """Check if the new_data path exists

  Args:
      path (str, optional): Path to the "new_data" folder. Defaults to "new_data".

  Returns:
      str: the path to the new_data folder
  """
  if not os.path.isdir(path):
      os.makedirs(path)
  return path

def cosine_similarity(v1: np.array, v2: np.array) -> float:
  """Calculate the cosine similarity between two vectors

  Args:
      v1 (np.array): one vector
      v2 (np.array): another vector

  Returns:
      float: similarity score between the two vectors
  """
  v1 = np.array(v1).flatten()
  v2 = np.array(v2).flatten()
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def return_emb(word: str, source_emb: pd.DataFrame) -> str or np.NaN:
  """Return the word that exists in the embedding

  Args:
      word (str): given word from game
      source_emb (pd.DataFrame): embedding dataset

  Returns:
      str or np.NaN: the filtered word that exists in the embedding dataset
  """
  try:
    emb = source_emb[word]
    return word
  except KeyError:
    word_stem = ps.stem(word)
    try:
      emb = source_emb[word_stem]
      return word_stem
    except KeyError:
      return np.NaN

def load_online_data(out_path: str, online_path:str = "./data/online.csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Load the data from the online dataset.

  Args:
      out_path (str): the path that the upgraded online dataset should be saved to.
      online_path (str, optional): the path where the online data exists. Defaults to "./data/online.csv".

  Returns:
      tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: the different versions of the online dataset
  """
  rankings = {'Easy': 1, 'Medium': 2, 'Hard': 3}

  online = pd.read_csv(online_path)
  online['Ranking'] = online['Level'].apply(lambda x: rankings[x])
  online_clues = online.loc[:, ['wordpair_id', 'gameID', 'clueOption1', 'clueOption2', 'clueOption3', 'clueOption4', 'clueOption5', 'clueOption6', 'clueOption7', 'clueOption8', 'clueFinal', 'Acc', 'Level', 'Ranking']]
  online_unique_clues = online[['wordpair_id', 'gameID', 'clueOption1', 'clueOption2', 'clueOption3', 'clueOption4', 'clueOption5', 'clueOption6', 'clueOption7', 'clueOption8', 'target1', 'target2']]
  online_clues['Number of Clues'] = [len(online_unique_clues.loc[i,:].dropna().to_list()) for i in range(len(online_clues))]
  online_guesses = online.loc[:, ['wordpair_id', 'gameID', 'GuessOption1', 'GuessOption2', 'GuessOption3', 'GuessOption4', 'GuessOption5', 'GuessOption6', 
                                  'GuessOption7', 'GuessOption8', "target1", "target2"]]
  online.to_csv(os.path.join(out_path, "candidate-generation.csv"), index = False)
  return (online, online_clues, online_unique_clues, online_guesses)

def load_embeddings_data(swow_path:str = "./data/swow_embeddings.csv") -> pd.DataFrame:
  """Load the SWOW embedding

  Args:
      swow_path (str, optional): path to the SWOW embeddings. Defaults to "./data/swow_embeddings.csv".

  Returns:
      pd.DataFrame: the read-in SWOW embeddings
  """
  return pd.read_csv(swow_path)

new_data_path = create_new_data_path()
online, online_clues, online_unique_clues, online_guesses = load_online_data(out_path=new_data_path)
swow = load_embeddings_data()

# We evaluate whether the responses show signatures of clustering and/or foraging typically found in semantic retrieval tasks. we use a patchy semantic space and ask whether the candidate responses show any evidence of transitions within and outside the patch, and whether these are related to correct responses from the listener. 
class PatchForaging:
  def __init__(self, online: pd.DataFrame = online, patch_name:str = "in_out_transitions.csv", movement_name: str = "patch_words.csv"):
    """Initialize variables and run methods to create data

    Args:
        online (pd.DataFrame, optional): online data. Defaults to online.
        patch_name (str, optional): name of the patch data file to be saved. Defaults to "in_out_transitions.csv".
        movement_name (str, optional): name of the movement data file to be saved. Defaults to "patch_words.csv".
    """
    self.patch_path = os.path.join(new_data_path, patch_name)
    self.movement_path = os.path.join(new_data_path, movement_name)
    self.online = online
    self.patch = self._patch()
    self.movement = self._determine_movement()

  def _patch(self) -> pd.DataFrame:
    """Determine the patch of words for each unique wordpair_id.

    Returns:
        pd.DataFrame: A dataframe of the patches with respect to the gameID and wordpair_id.
    """
    if os.path.exists(self.patch_path):
      return pd.read_csv(self.patch_path)

    wpid = self.online['wordpair_id'].unique()
    rows = []
    for t in wpid:
      temp = self.online[self.online['wordpair_id'] == t].reset_index()
      temp = temp.loc[:, ['wordpair_id', 'gameID', 'clueOption1', 'target1', 'target2', 'Level', 'Acc', 'clueFinal']].reset_index(drop = True)
      row_frame = temp[['gameID', 'clueFinal', 'Acc', 'Level']]
      row_frame['wordpair_id'] = t
      row_frame['target1'] = temp.loc[0, 'target1']
      row_frame['target2'] = temp.loc[0, 'target2']
      clues = list(temp['clueOption1'].unique())
      row_frame['words_in_patch'] = ','.join(clues)
      row_frame['patchsize'] = len(clues)
      row_frame['Level'] = temp.loc[0, 'Level']

      rows.append(row_frame)

    frame = pd.concat(rows).reset_index(drop = True).rename({'index': 'row_id'}, axis = 1)
    frame.to_csv(self.patch_path, index = False)
    return frame

  def _movement(self, words: list) -> str:
    """Method to determine the movement from the first word to the second and if it is
    inside the patch or not.

    Args:
        words (list): The two words to be compared for patch movement

    Returns:
        str: The movement designation.
    """
    first, second = words
    if first[1] == 1 and second[1] == 1:
      return 'In-In'
    elif first[1] == 1 and second[1] == 0:
      return 'In-Out'
    elif first[1] == 0 and second[1] == 1:
      return 'Out-In'
    else:
      return 'Out-Out'

  def _determine_movement(self) -> pd.DataFrame:
    """Method to determine the movement of the words inside of the patch and how many times each
    movement occurs.

    Returns:
        pd.DataFrame: A dataframe that contains the counts of the movements given the unique
        identifiers.
    """
    if os.path.exists(self.movement_path):
      return pd.read_csv(self.movement_path)
    
    temp_options = self.online[['wordpair_id', 'gameID', 'clueOption1', 'clueOption2', 'clueOption3','clueOption4', 'clueOption5', 'clueOption6', 'clueOption7', 'clueOption8', 'clueFinal', 'Acc']]
    temp_options = temp_options.reset_index().rename({'index': 'row_id'}, axis = 1)
    rows = []
    for i in tqdm(range(len(temp_options))):
      r = {}
      row = temp_options.loc[i, :].dropna()
      r['wordpair_id'] = row.wordpair_id
      r['gameID'] = row.gameID
      r['In-In'] = 0
      r['In-Out'] = 0
      r['Out-In'] = 0
      r['Out-Out'] = 0

      words = list(row.drop(['wordpair_id', 'Acc', 'clueFinal']).to_list())
      words = [w for w in words if isinstance(w, str)]
      patch_wordpair = self.patch.loc[self.patch['wordpair_id'] == row.wordpair_id, :]
      words_in_patch = [(w, 1) if w in patch_wordpair['words_in_patch'].to_list()[0].split(',') else (w, 0) for w in words ]
      perms = [(words_in_patch[i], words_in_patch[i+1]) for i in range(0, len(words_in_patch)-1)]
      movements = pd.Series([self._movement(p) for p in perms])
      ps = pd.DataFrame(movements.value_counts().reset_index()).rename({'index': 'Type', 0: 'Count'}, axis = 1)

      ps_d = dict(zip(ps['Type'], ps['Count']))
      for k, v in ps_d.items():
        r[k] = v
      r['Acc'] = row.Acc
      r['ClueFinal'] = row.clueFinal
      rows.append(r)
      
    total = pd.DataFrame.from_dict(rows)
    total.to_csv(self.movement_path, index = False)
    return total

# For each sequence of candidates generated by a person (lighter-flash-bright-lightning), 
# find out whether they are closer to one word (quick) or another (glow) and assign a 1 
# or 2 to each candidate response (use swow embeddings for determining semantic similarity)
class Predictions:
  swow_similarity: pd.DataFrame = None

  def __init__(self, 
               processed_path: str = new_data_path,
               clue: pd.DataFrame = online_unique_clues):
    """Init variables and run the methods to extract the data.

    Args:
        processed_path (str, optional): path for where the new data should be saved. Defaults to new_data_path.
        clue (pd.DataFrame, optional): dataset that contains all unique clues from the experiment. Defaults to online_unique_clues.
    """
    self.processed_path = processed_path
    self.clue = clue
    self.melted_clue = self._melt_clue()
    self.embeddings = self._create_embeddings()

    if Predictions.swow_similarity is None:
      Predictions.swow_similarity = self.swow_similarity_clustering()

  def _melt_clue(self) -> pd.DataFrame:
    """Melt or pivot longer the clue dataset

    Returns:
        pd.DataFrame: Melted dataframe
    """
    melted_clue = pd.melt(self.clue, id_vars = ['wordpair_id', 'gameID', 'target1', 'target2']).sort_values(['target1', 'target2']).dropna().reset_index(drop = True)
    melted_clue['value'] = melted_clue['value'].str.lower()
    return melted_clue

  def _create_embeddings(self) -> pd.DataFrame:
    """Extract the embeddings for each target (from wordpair_id) and given clue for eventual similarity measure between each.

    Returns:
        pd.DataFrame: Dataframe that contains all of the embeddings 
    """
    all_words = pd.Series(self.melted_clue['target1'].to_list() + self.melted_clue['target2'].to_list() + self.melted_clue['value'].to_list())
    unq_words = list(set(all_words))
    rows = []
    index_ = []
    none_vector = [np.NaN for _ in range(300)]
    for word in unq_words:

      # collect the embeddings for the targets and the given word
      try:
        emb = swow[return_emb(word, swow)].values.tolist()
      except:
        emb = none_vector

      rows.append(emb)
      index_.append(word)
    emb_data = pd.DataFrame(rows, index = index_)
    emb_data = emb_data.reset_index().rename({'index': 'value'}, axis = 1)
    return emb_data

  def swow_similarity_clustering(self) -> pd.DataFrame:
    """Establish clusters towards either target given the clue through cosine similarity
    from each words embeddings.

    Returns:
        pd.DataFrame: Predictions of which clue should be clustered into which target cluster
    """
    swow_path = os.path.join(self.processed_path, 'swow_arc.csv')
    if os.path.exists(swow_path):
      return pd.read_csv(swow_path)

    swow_data = self.melted_clue.copy()
    closesttarget = []
    for i, row in tqdm(self.melted_clue.iterrows()):
      target1, target2, word = row.target1, row.target2, row.value
      
      # extract words and check if the word is None
      target1 = self.embeddings[self.embeddings.value == target1].drop('value', axis = 1).values
      target2 = self.embeddings[self.embeddings.value == target2].drop('value', axis = 1).values
      word = self.embeddings[self.embeddings.value == word].drop('value', axis = 1).values

      if word is not None:
        sim_target1_word = cosine_similarity(target1, word)
        sim_target2_word = cosine_similarity(target2, word)
      else:
        sim_target1_word = sim_target2_word = 0

      # find similarity
      if sim_target1_word > sim_target2_word:
        closesttarget.append(1)
      else:
        closesttarget.append(2)

    swow_data['Prediction_SWOW'] = closesttarget
    swow_data = swow_data.sort_values(['wordpair_id', 'gameID', 'variable']).reset_index(drop = True)
    swow_data.to_csv(swow_path, index = False)
    return swow_data

pf = PatchForaging()
predictions = Predictions()