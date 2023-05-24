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
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <ctime>

using std::vector;
using std::string;
using std::fstream;
using std::cout;
using std::getline;
using std::stoi;
using std::stof;
using std::istringstream;
using std::to_string;
using std::mt19937;
using std::uniform_real_distribution;
using std::exp;
using std::round;
using std::pair;
using std::srand;
using std::rand;

const int COLUMNS_NUM = 10;
const int ROWS_NUM = 16;
const int RANDOM_STATE = 42;
const float MIN_WEIGHT_VALUE = -1.0;
const float MAX_WEIGHT_VALUE = 1.0;
const float BIAS_INITIAL_VALUE = -1.8;
const float LEARNING_RATE = 0.03;
const int ITERATION_NUMBER = 500;

mt19937 generator(RANDOM_STATE);
uniform_real_distribution<> dist(MIN_WEIGHT_VALUE, MAX_WEIGHT_VALUE);

// Valores esperados
vector<int> aExpected = {1, 0, 0, 0, 0};
vector<int> eExpected = {0, 1, 0, 0, 0};
vector<int> iExpected = {0, 0, 1, 0, 0};
vector<int> oExpected = {0, 0, 0, 1, 0};
vector<int> uExpected = {0, 0, 0, 0, 1};

class FileManager {
    public:
        fstream file;
        string mode;
        string filename;

        FileManager(string aFilename, string aMode) {
            filename = aFilename;
            mode = aMode;
            get_file();
        };

        vector<vector<float>> parse_float_matrix() {
            vector<vector<float>> matrix;
            string str_line;
            for(int i = 0; i < 16; i++) {
                getline(file, str_line);
                vector<float> row = {};
                istringstream iss(str_line);
                string token;

                while (iss >> token) {
                    row.push_back(stof(token));
                };

                matrix.push_back(row);
            }
            return matrix;
        };

        vector<vector<int>> parse_integer_matrix() {
            vector<vector<int>> matrix;
            string str_line;
            for(int i = 0; i < 16; i++) {
                getline(file, str_line);
                vector<int> row = {};
                istringstream iss(str_line);
                string token;

                while (iss >> token) {
                    row.push_back(stoi(token));
                };

                matrix.push_back(row);
            }
            return matrix;
        };

        float parse_float() {
            float number;
            string str_line;
            getline(file, str_line);
            number = stof(str_line);

            return number;
        }

        void get_file() {
            if (mode == "write") {
                file.open(filename, std::ios::out);
            } else {
                file.open(filename, std::ios::in);
            }
            
        };

        void write(string words) {
            file << words;
        };

        void line_break() {
            string str_line;
            getline(file, str_line);
        }
};

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
        vector<vector<float>> weights;
        float learningRate;

        /**
         * @brief Constructor de la clase Perceptron.
         * 
         * @param aLearningRate Parámetro de tipo flotante necesario para otorgar la razón de aprendizaje de la neurona. 
         * @param aRandomState Parámetro de tipo entero necesario para otorgar la semilla de aleatoriedad para generar 
         * los pesos de la neurona.
         
        */
        Perceptron(float aLearningRate, int aRandomState) {
            learningRate = aLearningRate;
            randomState = aRandomState;
            bias = BIAS_INITIAL_VALUE;
            initialize_weights();
        };

        /**
         * @brief Método utilizado para calcular la suma ponderada en base a las entradas y los pesos de la neurona. 
         * Se utiliza un producto punto entre el vector de entradas y el vector de pesos y, al final, se le suma el
         * sesgo (o bias). Fórmula: sum(x * w) + bias
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros (matriz de enteros) que hace referencia 
         * a los valores de entrada de la neurona. 
        */
        float net_input(vector<vector<int>> inputValues) {
            float result = 0;
            for (int i = 0; i < ROWS_NUM; i++) {
                for (int j = 0; j < COLUMNS_NUM; j++) {
                    result += inputValues[i][j] * weights[i][j];
                }
            }
            return result + bias;
        };

        /**
         * @brief Método utilizado para describir la función de activación de la neurona (función sigmoide). 
         * Se utiliza para determinar si las entradas son capaces de activar (excitar) o no (inhibir) a la neurona.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros (matriz de enteros) que hace referencia 
         * a los valores de entrada de la neurona.
        */
        float activation_function(vector<vector<int>> inputValues) {
            float weightedSum = net_input(inputValues);
            return 1 / (1 + exp(-weightedSum));
        };

        // AJUSTES DE LA NEURONA

        /**
         * @brief Método utilizado para ajustar los pesos del perceptron (neuorna) a través de la Regla Delta. 
         * Fórmula: w + L(s - y)x
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros (matriz de enteros) que hace referencia 
         * a los valores de entrada de la neurona.
         * @param expectedValue Parámetro de tipo entero que hace referencia al valor esperado en la salida de
         * la neurona.
         * @param outputValue Parámetro de tipo número de coma flotante que hace referencia al valor obtenido 
         * en la salida de la neurona.
        */
        void adjust_weights(vector<vector<int>> inputValues, int expectedValue, int outputValue) {
            for (int i = 0; i < ROWS_NUM; i++) {
                for (int j = 0; j < COLUMNS_NUM; j++) {
                    weights[i][j] += learningRate * (expectedValue - outputValue) * inputValues[i][j];
                };
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
        void adjust_bias(int expectedValue, int outputValue) {
            bias += learningRate * (expectedValue - outputValue) * 1;
        };

        // MÉTODOS DE UTILIDAD DEL PERCEPTRON

        /**
         * @brief Método utilizado para inicializar el vector de pesos de la neurona aleatoriamente.
         * 
         * @param weightsNumber Parámetro de tipo entero que determina el número de pesos que serán 
         * incializados para la neurona. 
        */
        void initialize_weights() {
            for (int i = 0; i < ROWS_NUM; i++) {
                vector<float> new_row = {};
                for (int i = 0; i < COLUMNS_NUM; i++) {
                    new_row.push_back(dist(generator));
                };
                weights.push_back(new_row);
            };
        };

        /**
         * @brief Método utilizado para mostrar el vector de pesos de la neurona en la consola.
        */
        void show_weights() {
            for (vector<float> weight : weights) {
                for (float w : weight) {
                    cout << w << ", ";
                };
                cout << "\n";
            };
        };

        /**
         * @brief Método utilizado para retornar el vector de pesos de la neurona en una cadena de caracteres.
        */
        string weights_to_string() {
            string result = "";
            for (vector<float> weight : weights) {
                string row = "";
                for (float w : weight) {
                    row += to_string(w) + " ";
                };
                row = row.substr(0, row.length() - 1);
                result += row + "\n";
            };
            string bias_str = to_string(bias);
            result += bias_str.substr(0, bias_str.length() - 4) + "\n\n";
            return result;
        };

        // SETTERS

        /**
         * @brief Método utilizado para asignar un nuevo vector de pesos a la neurona.
        */
        void set_weights(vector<vector<float>> newWeights) {
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
                Perceptron newPerceptron(LEARNING_RATE, 42);
                perceptrons.push_back(newPerceptron);
            };
            iterationsNumber = aIterationsNumber;
        }

        /**
         * @brief Método utilizado para procesar un patrón específico con la red neuronal. Este método utiliza la Regla Delta 
         * para cambiar los pesos y el sesgo (o bias) de cada uno de sus perceptrones (neuornas), si y solo si, la diferencia 
         * entre el valor esperado y el valor obtenido es diferente de 0.
         * 
         * @param inputValues Parámetro de tipo vector de vectores de enteros que representa la entrada de la red neuronal.
         * @param expectedValues Parámetro de tipo vector de enteros que representa la salida esperada de la red neuronal.
        */
        vector<float> process_input(vector<vector<int>> inputValues, vector<int> expectedValues) {
            vector<float> output = {};
            for (int i = 0; i < perceptrons.size(); i++) {
                float predict = perceptrons[i].activation_function(inputValues);
                output.push_back(predict);
            };
            return output;
        };

        /**
         * @brief Método utilizado para entrenar a la red neuronal con un conjunto de patrones. Durante un número de iteraciones
         * determinado, se eligirá aleatoriamente el patrón con el que se desea entrenar y la red neuronal empezará a procesarlo con 
         * todas los perceptrones (neuronas); reajustando sus pesos cuando sea necesario.
         * 
         * @param patterns Parámetro de tipo vector de pares de matrices y vectores utilizado para representar una entrada
         * y su salida esperada.
        */
        void training(vector<pair<vector<vector<int>>, vector<int>>> patterns) {
            int n = patterns.size();
            for (int iteration = 0; iteration < iterationsNumber; iteration++) {
                int index = rand() % n;
                vector<vector<int>> input = patterns[index].first;
                vector<int> expectedValue = patterns[index].second;
                vector<float> output = process_input(input, expectedValue);
                
                for (int i = 0; i < output.size(); i++) {
                    int outputValue = round(output[i]);
                    int error = expectedValue[i] - outputValue;
                    if (error != 0) {
                        perceptrons[i].adjust_weights(input, expectedValue[i], outputValue);
                        perceptrons[i].adjust_bias(expectedValue[i], outputValue);
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
            vector<float> output = {};
            for (int i = 0; i < perceptrons.size(); i++) {
                float result = perceptrons[i].activation_function(inputValues);
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
            float min_output = 0.3;
            for (int i = 0; i < results.size(); i++) {
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
            vector<char> letters = {'a', 'e', 'i', 'o', 'u'};
            for (int i = 0; i < output.size(); i++) {
                if (output[i] == 1) {
                   cout << "Esto es una " << letters[i];
                };
            };
        };

        // KNOWLEDGE BASE

        void import_knowledge_base() {
            FileManager fileManager("base.txt", "read");
            for (Perceptron &perceptron : perceptrons) {
                vector<vector<float>> weights = fileManager.parse_float_matrix();
                float bias = fileManager.parse_float();
                fileManager.line_break();

                perceptron.set_weights(weights);
                perceptron.set_bias(bias);
            }
        };

        void export_knowledge_base() {
            FileManager fileManager("base.txt", "write");

            for (Perceptron perceptron : perceptrons) {
                string perceptron_info = perceptron.weights_to_string();
                fileManager.write(perceptron_info);
            };
        };
};

/*
vector<pair<vector<vector<int>>, vector<int>>> get_patterns() {
    return nullptr;
};
*/

// CÓDIGO PARA TESTING DE LA RED NEURONAL

void resolve_network(NeuralNetwork neural, vector<vector<vector<int>>> inputVector) {
    //neural.show_neural_weights();
    for (int i = 0; i < inputVector.size(); i++) {
        neural.show_results(neural.resolve(inputVector[i]));
        cout << "\n";
    };
};

int main(){
    // La forma más engorrosa de determinar los números aleatorios
    srand(static_cast<unsigned int>(std::time(nullptr)));
    
    NeuralNetwork neural(5, ITERATION_NUMBER);
    
    vector<vector<int>> inputA_1 = {
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

    vector<vector<int>> inputA_2 = {
        {0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,1,1,1,1,0,0,0},
{0,0,1,1,1,1,1,1,0,0},
{0,1,1,1,0,0,1,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,1,0,0,1,1,1,0},
{0,0,1,1,1,1,1,1,1,0},
{0,0,0,1,1,1,1,0,1,1},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0}
    };

    vector<vector<int>> inputE_1 = {
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

    vector<vector<int>> inputE_2 = {
        {0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,1,1,1,1,0,0,0},
{0,0,1,1,1,1,1,1,0,0},
{0,0,1,1,0,0,1,1,0,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,1,1,1,1,1,1,0},
{0,1,1,0,0,0,0,0,0,0},
{0,1,1,0,0,0,0,0,0,0},
{0,1,1,1,0,0,0,0,0,0},
{0,0,1,1,1,1,1,1,0,0},
{0,0,0,1,1,1,1,1,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0}
    };

    vector<vector<int>> inputI_1 = {
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

    vector<vector<int>> inputI_2 = {
        {0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0}
    };

    vector<vector<int>> inputO_1 = {
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
{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<int>> inputO_2 = {
        {0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,1,1,1,1,0,0,0},
{0,0,1,1,1,1,1,1,0,0},
{0,1,1,1,0,0,1,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,1,0,0,1,1,1,0},
{0,0,1,1,1,1,1,1,0,0},
{0,0,0,1,1,1,1,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0}
    };

    vector<vector<int>> inputU_1 = {
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

    vector<vector<int>> inputU_2 = {
        {0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,0,0,0,0,1,1,0},
{0,1,1,1,0,0,1,1,1,0},
{0,0,1,1,1,1,1,1,1,0},
{0,0,0,1,1,1,1,0,1,1},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0}
    };

    vector<vector<int>> testU = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, 
{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, 
{0, 0, 1, 0, 0, 0, 0, 0, 1, 0}, 
{0, 0, 1, 0, 0, 0, 0, 1, 1, 0}, 
{0, 0, 1, 1, 0, 0, 0, 1, 1, 1}, 
{0, 0, 0, 1, 1, 1, 1, 1, 0, 1}, 
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    pair<vector<vector<int>>, vector<int>> a_1(inputA_1, aExpected);
    pair<vector<vector<int>>, vector<int>> e_1(inputE_1, eExpected);
    pair<vector<vector<int>>, vector<int>> i_1(inputI_1, iExpected);
    pair<vector<vector<int>>, vector<int>> o_1(inputO_1, oExpected);
    pair<vector<vector<int>>, vector<int>> u_1(inputU_1, uExpected);

    pair<vector<vector<int>>, vector<int>> a_2(inputA_2, aExpected);
    pair<vector<vector<int>>, vector<int>> e_2(inputE_2, eExpected);
    pair<vector<vector<int>>, vector<int>> i_2(inputI_2, iExpected);
    pair<vector<vector<int>>, vector<int>> o_2(inputO_2, oExpected);
    pair<vector<vector<int>>, vector<int>> u_2(inputU_2, uExpected);

    vector<pair<vector<vector<int>>, vector<int>>> patterns = {a_1, e_1, i_1, o_1, u_1, a_2, e_2, i_2, o_2, u_2};

    vector<vector<vector<int>>> inputVector = {inputA_1, inputE_1, inputI_1, inputO_1, inputU_1};
    //neural.training(patterns);
    //neural.export_knowledge_base();
    neural.import_knowledge_base();

    /*
    for (int i = 0; i < 5; i++) {
        cout << neural.perceptrons[i].weights_to_string();
    }
    */

    resolve_network(neural, inputVector);

    return 0;
};