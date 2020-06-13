import numpy
from keras import Sequential, Input
from keras.layers import Dense

from matplotlib import pyplot

from keras import backend as K

def main():
    seq = Sequential()
    seq.add(Dense(units=2, activation='tanh'))
    seq.add(Dense(units=1, activation='tanh'))

    seq.compile(optimizer='rmsprop', loss='mean_squared_error')

    input_data = numpy.array(
            [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
    )
    normalized_input_data = input_data / 12

    output_data = numpy.array(
            [[0], [2], [6], [9], [14], [16], [19], [18], [15], [11], [5], [2]]
    )
    normalized_output_data = (output_data - 10) / 15

    seq.fit(normalized_input_data, normalized_output_data, epochs=5000)

    normalized_predicted = seq.predict(normalized_input_data)
    predicted = normalized_predicted * 15 + 10
    print(predicted)

    pyplot.plot(input_data, output_data)
    pyplot.plot(input_data, predicted)
    pyplot.show()


if __name__ == '__main__':
    main()
