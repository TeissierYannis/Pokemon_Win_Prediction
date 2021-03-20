import pandas as pnd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib


# setup data
def setup_data():
    pnd.set_option('display.max_columns', None)
    pnd.set_option('mode.chained_assignment', None)

    # Dataframe
    pokemons = pnd.read_csv('data/pokedex.csv', sep=',', encoding='latin-1')

    # transform legendary column to int
    pokemons['LEGENDAIRE'] = (pokemons['LEGENDAIRE'] == 'VRAI').astype(int)

    # load fights
    fights = pnd.read_csv('data/combats.csv', sep=',', encoding='latin-1')

    nbFirstPosition = fights.groupby('Premier_Pokemon').count()
    nbSecondPosition = fights.groupby('Second_Pokemon').count()
    nbVictories = fights.groupby('Pokemon_Gagnant').count()

    aggregation = fights.groupby('Pokemon_Gagnant').count()
    aggregation.sort_index()

    aggregation['NBR_COMBATS'] = nbFirstPosition.Pokemon_Gagnant + nbSecondPosition.Pokemon_Gagnant
    aggregation['NB_VICTOIRES'] = nbVictories.Premier_Pokemon

    # % of victory
    aggregation['POURCENTAGE_DE_VICTOIRE'] = nbVictories.Premier_Pokemon / (
            nbFirstPosition.Pokemon_Gagnant + nbSecondPosition.Pokemon_Gagnant)

    newPokedex = pokemons.merge(aggregation, left_on='NUMERO', right_index=True, how='left')

    dataset = newPokedex
    dataset = dataset.dropna(axis=0, how='any')
    dataset.to_csv('data/dataset.csv', sep='\t')
    return dataset


# train and save model
def learn_and_save(dataset):
    # NIVEAU_ATTAQUE;NIVEAU_DEFFENSE;NIVEAU_ATTAQUE_SPECIALE;NIVEAU_DEFENSE_SPECIALE;VITESSE;NOMBRE_GENERATIONS
    X = dataset.iloc[:, 4:11].values
    Y = dataset.iloc[:, 16].values

    X_LEARN, X_VALIDATE, Y_LEARN, Y_VALIDATE = train_test_split(X, Y, test_size=0.2, random_state=0)

    algo = RandomForestRegressor()
    algo.fit(X_LEARN, Y_LEARN)
    predictions = algo.predict(X_VALIDATE)
    precision = r2_score(Y_VALIDATE, predictions)
    precision_learn = algo.score(X_LEARN, Y_LEARN)
    print("=========== RANDOM FOREST REGRESSION ==========")
    print("Precision Learn : " + str(precision_learn))
    print("Precision Validation : " + str(precision))
    print("===============================================")

    file = 'models/model_pokemon.mod'
    joblib.dump(algo, file)


# setup and train model
learn_and_save(setup_data())
