/**
 * PROYECTO I - RED NEURONAL PARA DIFERENCIAR LETRAS VOCALES MANUSCRITAS
 * Hecha por: Kevin Rojas - C.I 29.582.382
 * Materia: Modelos de Programación Emergente (MPE)
 *   
 * Enunciado: Desarrolle una RNA basada en el Perceptron Simple que permita el reconocimiento 
 * de letras minúsculas (vocales: a, e, i, o, u) manuscritas. La entrada será un archivo contentivo 
 * del caracter fotografiado, o una matriz 16x10 que simule el caracter, a analizar.
*/
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>

using std::vector;
using std::cout;
using std::string;
using std::mt19937;
using std::uniform_real_distribution;
using std::exp;
using std::round;

const float MIN_WEIGHT_VALUE = -1.0;
const float MAX_WEIGHT_VALUE = 1.0;
const float BIAS_INITIAL_VALUE = -1.8;
const float LEARNING_RATE = 0.05;
const int ITERATION_NUMBER = 500;

// Valores esperados
vector<int> aExpected = {1, 0, 0, 0, 0};
vector<int> eExpected = {0, 1, 0, 0, 0};
vector<int> iExpected = {0, 0, 1, 0, 0};
vector<int> oExpected = {0, 0, 0, 1, 0};
vector<int> uExpected = {0, 0, 0, 0, 1};

/**
 * @brief Clase Perceptron que modela la neurona propuesta por Frank Rosenblatt.
 * 
 * Se utiliza como una neuorna individual en una red neuronal. Posee N número de entradas (dendritas), con
 * sus respectivos pesos, una razón de aprendizaje, la sumatoria ponderada y una función de activación.   
*/
class Perceptron {
    public:
        int randomState;
        float bias;
        vector<float> weights;
        float learningRate;

        /**
         * @brief Constructor de la clase Perceptron.
         * 
         * @param aLearningRate Parámetro de tipo flotante necesario para otorgar la razón de aprendizaje de la neurona. 
         * @param aRandomState Parámetro de tipo entero necesario para otorgar la semilla de aleatoriedad para generar 
         * los pesos de la neurona.
         * @param weightsNumber Parámetro de tipo entero necesario para determinar el número de entradas (dendritas) 
         * y pesos asociados a la neurona. 
        */
        Perceptron(float aLearningRate, int aRandomState, int weightsNumber) {
            learningRate = aLearningRate;
            randomState = aRandomState;
            bias = BIAS_INITIAL_VALUE;
            initialize_weights(weightsNumber);
        };

        /**
         * @brief Método utilizado para calcular la suma ponderada en base a las entradas y los pesos de la neurona. 
         * Se utiliza un producto punto entre el vector de entradas y el vector de pesos y, al final, se le suma el
         * sesgo (o bias). Fórmula: sum(x * w) + bias
         * 
         * @param inputValues Parámetro de tipo vector de enteros que hace referencia a los valores de entrada de 
         * la neurona. 
        */
        float net_input(vector<int> inputValues) {
            float result = 0;
            for (int index = 0; index < inputValues.size(); index++) {
                result += inputValues.at(index) * weights.at(index);
            }
            return result + bias;
        };

        /**
         * @brief Método utilizado para describir la función de activación de la neurona (función sigmoide). 
         * Se utiliza para determinar si las entradas son capaces de activar (excitar) o no (inhibir) a la neurona.
         * 
         * @param inputValues Parámetro de tipo vector de enteros que hace referencia a los valores de entrada de 
         * la neurona. 
        */
        float activation_function(vector<int> inputValues) {
            float weightedSum = net_input(inputValues);
            return 1 / (1 + exp(-weightedSum));
        };

        // AJUSTES DE LA NEURONA

        /**
         * @brief Método utilizado para ajustar los pesos del perceptron (neuorna) a través de la Regla Delta. 
         * Fórmula: w + L(s - y)x
         * 
         * @param inputValues Parámetro de tipo vector de enteros que hace referencia a los valores de entrada de 
         * la neurona.
         * @param expectedValue Parámetro de tipo entero que hace referencia al valor esperado en la salida de
         * la neurona.
         * @param outputValue Parámetro de tipo número de coma flotante que hace referencia al valor obtenido 
         * en la salida de la neurona.
        */
        void adjust_weights(vector<int> inputValues, int expectedValue, float outputValue) {
            for (int i = 0; i < weights.size(); i++) {
                weights[i] = weights[i] + (learningRate * (expectedValue - outputValue) * inputValues[i]);
            };
        };

        /**
         * @brief Método utilizado para ajustar el sesgo (o bias) del perceptron (neuorna) a través de la Regla Delta. 
         * Fórmula: b + L(s - y)1
         * 
         * @param expectedValue Parámetro de tipo entero que hace referencia al valor esperado en la salida de
         * la neurona.
         * @param outputValue Parámetro de tipo número de coma flotante que hace referencia al valor obtenido 
         * en la salida de la neurona.
        */
        void adjust_bias(int expectedValue, float outputValue) {
            bias = bias + (learningRate * (expectedValue - outputValue) * 1);
        };

        // MÉTODOS DE UTILIDAD DEL PERCEPTRON

        /**
         * @brief Método utilizado para inicializar el vector de pesos de la neurona aleatoriamente.
         * 
         * @param weightsNumber Parámetro de tipo entero que determina el número de pesos que serán 
         * incializados para la neurona. 
        */
        void initialize_weights(int weightsNumber) {
            mt19937 generator(randomState);
            uniform_real_distribution<> dist(MIN_WEIGHT_VALUE, MAX_WEIGHT_VALUE);
            for (int i = 0; i < weightsNumber; i++) {
                weights.push_back(dist(generator));
            };
        };

        /**
         * @brief Método utilizado para mostrar el vector de pesos de la neurona en la consola.
        */
        void show_weights() {
            for (float weight : weights) {
                cout << weight << ", ";
            };
        };

        // SETTERS

        /**
         * @brief Método utilizado para asignar un nuevo vector de pesos a la neurona.
        */
        void set_weights(vector<float> newWeights) {
            weights = newWeights;
        };

        /**
         * @brief Método utilizado para asignar un nuevo sesgo (o bias) a la neurona.
        */
        void set_bias(float newBias) {
            bias = newBias;
        };
};

/**
 * @brief Clase NeuralNetwork que modela una red neuronal de N perceptrones simples.
 * 
 * Se utiliza para analizar el problema planteado en el enunciado: identificar o reconocer
 * letras vocales minúsculas (a, e, i, o, u), a través de una matriz de entrada. La salida 
 * de la red neuronal es un vector de enteros de tamaño N.    
*/
class NeuralNetwork {
    public:
        vector<Perceptron> perceptrons;
        int iterationsNumber;

        /**
         * @brief Constructor de la clase NeuralNetwork.
         * 
         * @param aIterationsNumber Parámetro de tipo entero que determina el número de iteraciones que hará la red neuronal
         * para entrenar y aprender los patrones.
         * @param aPerceptronsNumber Parámetro de tipo entero necesario para determinar el número de perceptrones (neuronas) 
         * en la red neuronal. 
        */
        NeuralNetwork(int aPerceptronsNumber, int aIterationsNumber) {
            for (int i = 0; i < aPerceptronsNumber; i++) {
                Perceptron newPerceptron(LEARNING_RATE, 42, 10);
                perceptrons.push_back(newPerceptron);
            };
            iterationsNumber = aIterationsNumber;
        }

        /**
         * @brief Método utilizado para entrenar a la red neuronal para aprender acerca de un patrón específico. Este utiliza
         * la Regla Delta para cambiar los pesos y el sesgo (o bias) de cada uno de sus perceptrones (neuornas), si y solo si, 
         * la diferencia entre el valor esperado y el valor obtenido es diferente de 0.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros que representa la entrada de la red neuronal.
         * @param expectedValues Parámetro de tipo vector de enteros que representa la salida esperada de la red neuronal.
        */
        void training(vector<vector<int>> inputValues, vector<int> expectedValues) {
            for (int iteration = 0; iteration < iterationsNumber; iteration++) {
                for (int i = 0; i < perceptrons.size(); i++) {
                    for (int j = 0; j < inputValues.size(); j++) {
                        float predict = perceptrons[i].activation_function(inputValues[j]);
                        int outputValue = round(predict);
                        int error = expectedValues[i] - outputValue;
                        if (error != 0) {
                            perceptrons[i].adjust_weights(inputValues[j], expectedValues[i], predict);
                            perceptrons[i].adjust_bias(expectedValues[i], predict);
                        };
                    };
                };
            };
        };

        /**
         * @brief Método utilizado para identificar o reconocer la matriz de entrada a través de la red neuronal 
         * previamente entrenada. Este devuelve un vector de tamaño N.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros que representa la entrada 
         * de la red neuronal.
        */
        vector<int> resolve(vector<vector<int>> inputValues) {
            vector<float> output;
            for (int i = 0; i < perceptrons.size(); i++) {
                float result = 0.0;
                for (int j = 0; j < inputValues.size(); j++) {
                    result = result + perceptrons[i].activation_function(inputValues[j]);
                };
                //float media = (result / 16);
                //cout << media << "\n";
                output.push_back(result);
            };
            return competition(output);
        };

        /**
         * @brief Método utilizado para determinar que perceptron debe responder de acuerdo a la competencia 
         * entre ellos. La respuesta debe sobrepasar un mínimo valor esperado y, luego, el perceptron con el 
         * mejor acercamiento responderá. 
         * 
         * @param results Parámetro de tipo vector de números de coma flotante que representa los acercamientos 
         * de cada neurona de la red neuronal.
        */
        vector<int> competition(vector<float> results) {
            vector<int> output = {0, 0, 0, 0, 0};
            int index = -1;
            float min_output = 1.0;
            for (int i = 0; i < results.size(); i++) {
                //cout << results[i] << "\n";
                if (results[i] > min_output) {
                    min_output = results[i];
                    index = i;
                };
            };

            if (index != -1) {
                output[index] = 1;
            }
            return output;
        };

        /**
         * @brief Método utilizado para mostrar el vector de pesos de cada neurona de red neuronal 
         * en la consola.
        */
        void show_neural_weights() {
            int count = 0;
            for (Perceptron perc : perceptrons) {
                cout << "Perceptron nº" << count << "\n";
                perc.show_weights();
                cout << "\n";
                cout << "Bias Perceptron nº" << count << ": " << perc.bias << "\n";
                count++;
            };
        };

        /**
         * @brief Método utilizado para mostrar el vector de respuesta de la red neuronal en la consola.
         * 
         * @param output Parámetro de tipo vector de enteros que representa la salida de la red neuronal.
        */
        void show_results(vector<int> output) {
            for (int out : output) {
                cout << out << " ";
            };
        };
};

// CÓDIGO PARA TESTING DE LA RED NEURONAL

void train_network(NeuralNetwork neural, int perceptronNumber, vector<vector<vector<int>>> inputVector) {
    vector<vector<int>> expectedVector = {aExpected, eExpected, iExpected, oExpected, uExpected};
    vector<char> letterVector = {'A', 'E', 'I', 'O', 'U'};
    for (int n = 0; n < perceptronNumber; n++) {
        cout << "ENTRENAMIENTO EN " << letterVector[n] << "\n";
        neural.training(inputVector[n], expectedVector[n]);
        neural.show_neural_weights();
        cout << "\n";
    }
    
};

void resolve_network(NeuralNetwork neural, vector<vector<vector<int>>> inputVector) {
    /*
    // Pesos entrenados
    vector<float> firstTrained = {0.593086, -0.179281, 1.09356, 0.590156, 0.208448, -0.483268, 0.314953, 0.201591, -0.126518, 0.399662};
    vector<float> secondTrained = {0.593086, -0.130374, 1.13431, 0.608691, 0.262472, -0.429244, 0.333488, 0.119234, -0.369221, 0.301777};
    vector<float> thirdTrained = {0.593086, -0.63313, 0.559382, 0.685451, 0.383416, -0.308299, 0.410249, -0.332583, -0.714266, 0.301777};
    vector<float> fourthTrained = {0.593086, -0.220064, 1.05057, 0.592189, 0.179093, -0.512622, 0.316987, 0.158607, -0.3012, 0.301777};
    vector<float> fifthTrained = {0.593086, -0.138074, 1.12086, 0.432708, 0.0947266, -0.596989, 0.157506, 0.2289, -0.0161487, 0.438412};

    // Bias entrenados
    float firstBias = 0.0131103;
    float secondBias = 0.0235552;
    float thirdBias = 0.0157665;
    float fourthBias = 0.0224136;
    float fifthBias = 0.0180892;

    neural.perceptrons[0].set_weights(firstTrained);
    neural.perceptrons[0].set_bias(firstBias);

    neural.perceptrons[1].set_weights(secondTrained);
    neural.perceptrons[1].set_bias(secondBias);

    neural.perceptrons[2].set_weights(thirdTrained);
    neural.perceptrons[2].set_bias(thirdBias);

    neural.perceptrons[3].set_weights(fourthTrained);
    neural.perceptrons[3].set_bias(fourthBias);

    neural.perceptrons[4].set_weights(fifthTrained);
    neural.perceptrons[4].set_bias(fifthBias);
    */
    for (int i = 0; i < inputVector.size(); i++) {
        neural.show_results(neural.resolve(inputVector[i]));
        cout << "\n";
    };
};

int main(){
    //Perceptron perceptron(LEARNING_RATE, 42, 10);
    NeuralNetwork neural(5, ITERATION_NUMBER);
    
    vector<vector<int>> inputA = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 1, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 1, 1},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<int>> inputE = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 1, 1, 1, 1, 1, 1, 1, 1, 0},
{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 1, 1, 1, 1, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
{0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
{0, 1, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<int>> inputI = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<int>> inputO = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<int>> inputU = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
{0, 0, 1, 1, 1, 1, 1, 1, 1, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 1, 1},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    int n = 5;
    vector<vector<vector<int>>> inputVector = {inputA, inputE, inputI, inputO, inputU};
    train_network(neural, n, inputVector);

    resolve_network(neural, inputVector);

    return 0;
};