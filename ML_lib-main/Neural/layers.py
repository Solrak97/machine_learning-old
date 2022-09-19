from matplotlib.pyplot import pink
from activation import sigmoid, d_sigmoid, relu, d_relu, tanh, d_tanh, lrelu, d_lrelu
import numpy as np

class DenseLayer:
    tipo = None
    neuronas = None
    activacion = None
    gradiente = None
    netos = None
    salida = None

    def __init__(self, neuronas=5, activacion=None, tipo="entrada"):
        self.tipo = tipo
        self.neuronas = neuronas
        self.activacion_str = activacion
        
        if self.tipo != 'entrada':
            self.activacion, self.gradiente = self.get_activation(activacion)
            self.activacion, self.gradiente = np.vectorize(self.activacion), np.vectorize(self.gradiente)
        pass

    def __str__(self) -> str:
        layer = f'''
        ==============================
        Tipo:           {self.tipo}
        Neuronas:       {self.neuronas}
        Activación:     {self.activacion_str}
        ==============================
        '''
        return layer

    def get_activation(self, f_name):
        if f_name == 'r':
            return (relu, d_relu)

        if f_name == 'l':
            return (lrelu, d_lrelu)

        if f_name == 't':
            return (tanh, d_tanh)

        if f_name == 's':
            return (sigmoid, d_sigmoid)

    def activate(self):
        if self.tipo == "entrada":
            self.salida = self.netos
            return self.netos

        else:
            self.salida = self.activacion(self.netos)
            return self.salida

    def set_netos(self, netos):
        self.netos = netos
        pass
