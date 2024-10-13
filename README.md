Parsear los argumentos (mode (train, test), config) y en el futuro poder modificar los argumentos desde la consola

Por ahora ignorare el debugging, concentrate solo en la funcionalidad

Cargar la configuracion desde el json a la clase DotDict y despues sobreescribir con los nuevos argumentos

Instanciar el DeviceSpec y usarlo with tf.device para seleccionar el algoritmo (alphazero, muzero) y llamar la funcion learn correspondiente

Instanciar AlphaZeroNet (DefaultAlphaZero), cargar el checkpoint si es requerido, instanciar AlphaZeroCoach, cargar los train_examples si es requerido y llamar la funcion learn

Empieza un ciclo con el numero de juegos (num_selfplay_iterations), despues otro ciclo (num_episodes) y para cada episodio clear_tree de mcts, ejecuto un episodio y guardas el GameHistory en el iteration_train_expample


Cambiar todos los Generic de typing por Union con sus posibles tipos



main
    learn_alphazero
        AlphaZeroNet
            AlphaZeroGymNetwork
        AlphaZeroCoach
            Coach
                AlphaZeroMCTS
                AlphaZeroPlayer


    learn_muzero
        MuZeroNet
        MuZeroCoach