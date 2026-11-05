import numpy as np

from tenzo.core.variable import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_cretor(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError
    
#helper function
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x