import csv

from sklearn.externals import joblib


# search pokemon information in pokedex
def find_pokemon_data(id_pokemon, pokedex):
    pokemon_data = []
    for pokemon in pokedex:
        if int(pokemon[0]) == id_pokemon:
            pokemon_data = [pokemon[0], pokemon[1], pokemon[4], pokemon[5], pokemon[6], pokemon[7], pokemon[8],
                            pokemon[9], pokemon[10]]
            break
    return pokemon_data


# predict victory percentage of pokemon with trained model
def predict(id_first_pokemon, id_second_pokemon, pokedex):
    first_pokemon = find_pokemon_data(id_first_pokemon, pokedex)
    second_pokemon = find_pokemon_data(id_second_pokemon, pokedex)

    predict_model = joblib.load('models/model_pokemon.mod')
    predict_first_pokemon = predict_model.predict(
        [[first_pokemon[2], first_pokemon[3], first_pokemon[4], first_pokemon[5],
          first_pokemon[6], first_pokemon[7], first_pokemon[8]]])
    predict_second_pokemon = predict_model.predict([[second_pokemon[2], second_pokemon[3], second_pokemon[4],
                                                     second_pokemon[5], second_pokemon[6], second_pokemon[7],
                                                     second_pokemon[8]]])

    print('(' + str(id_first_pokemon) + ') ' + first_pokemon[1] + ' VS ' + str(
        id_second_pokemon) + ') ' +
          second_pokemon[1])
    print('========== VICTORY RATES ==========')
    print(first_pokemon[1] + ' : ' + str(100 * predict_first_pokemon) + '%')
    print(second_pokemon[1] + ' : ' + str(100 * predict_second_pokemon) + '%')
    print('\n')
    if predict_first_pokemon > predict_second_pokemon:
        print(first_pokemon[1] + ' IS THE WIN !')
    else:
        print(second_pokemon[1] + ' IS THE WIN !')

# example
with open('data/pokedex.csv', newline='') as csvfile:
    pokedex = csv.reader(csvfile)
    next(pokedex)
    predict(10, 800, pokedex)
