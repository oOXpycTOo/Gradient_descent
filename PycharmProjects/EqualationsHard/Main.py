from sympy import Symbol, sympify, solve, lambdify, diff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import animation


# Класс для точки x0 (протестирую его позже)
class Point:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2


class GraphAnimation(animation.FuncAnimation):
    def __init__(self, x1, x2, func, fig=None, ax=None, frames=None, interval=60,
                 repeat_delay=5, blit=True, **kwargs):
        self.fig = fig
        self.ax = ax
        self.x1 = x1
        self.x2 = x2
        self.z = func(x1, x2)
        self.line, = self.ax.plot([], [], 'b', lw=2)
        self.point, = self.ax.plot([], [], 'bo')
        super(GraphAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                            frames=frames, interval=interval, blit=blit,
                                            repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        self.point.set_data([], [])
        self.point.set_3d_properties([])
        return self.line, self.point

    def animate(self, i):
        self.line.set_data(self.x1[:i], self.x2[:i])
        self.line.set_3d_properties(self.z[:i])
        self.point.set_data(self.x1[i - 1:i], self.x2[i - 1:i])
        self.point.set_3d_properties(self.z[i - 1:i])
        return self.line, self.point


class Frame:
    def __init__(self, x1, x2):
        self.right = np.array(x1).max() + 0.1*np.array(x1).max()
        self.left = np.array(x1).min() - 0.1*np.array(x1).min()
        self.up = np.array(x2).max() + 0.1*np.array(x2).max()
        self.down = np.array(x2).min() - 0.1*np.array(x2).min()
        self.step_w = 10 / np.array(x1).size
        self.step_h = 10 /np.array(x2).size


class GradientDescent:
    def __init__(self, x1, x2, function, tol=0.1, it_max=100):
        # sympify - функция перевода строки в выражение Sympy
        self.func = sympify(function)
        # создание объектов класса Symbol
        # грубо говоря, если у нас есть выражение 'x1+x2'
        # то для его написания и использования нужно будет писать:
        # self.x1_sym + self.x2_sym
        self.x1_sym = Symbol('x1')
        self.x2_sym = Symbol('x2')
        # присваивание численных переменных
        self.x1_num = x1
        self.x2_num = x2
        # создание объекта альфа класса Symbol
        self.alpha = Symbol('a')
        # создаём символьный градиент дифференцировнием функции func по
        # x1 и x2 (просто формула)
        self.grad_sym = [diff(function, self.x1_sym), diff(function, self.x2_sym)]
        # Создание численного градиента для начальной точки x0(x1, x2)
        # А это уже численное значение формулы, при подстановке туда х1, х2
        self.grad_num = [float(self.grad_sym[0].evalf(subs={self.x1_sym: x1, self.x2_sym: x2})),
                         float(self.grad_sym[1].evalf(subs={self.x1_sym: x1, self.x2_sym: x2}))]
        # Вычисление нормы матрицы, для проверки условия сходимости
        self.norm = np.sqrt(np.sqrt(float(self.grad_num[0]**2) + float(self.grad_num[1]**2)))
        # Максимальное число итерация(по умолчанию - 100)
        self.it_max = it_max
        # Точность, по умолчанию - 0.1
        self.tol = tol
        # Таблица со значениями
        self.data = {'x1': [self.x1_num],
                     'x2':[self.x2_num],
                     'Градиент':[self.grad_num],
                     'Итерация':[0],
                     'Норма матрицы':[self.norm],
                     'Альфа':[self.alpha]}
        self.it = 1
        # Погнали

    def gradient_descent(self):
        tol = self.tol
        alpha = self.alpha
        grad_sym = [self.grad_sym[0], self.grad_sym[1]]
        # Пока не превышено количество итераций
        # И пока точность плоха
        while self.it < self.it_max and self.norm > tol:
            # x1_1 = x1_0 + alpha*grad[0], где alpha - это пока что символьное значение, grad[0] - численное
            # x2_1 = x2_0 + alpha*grad[1], где alpha - это пока что символьное значение, grad[1] - численное
            x1_sym = self.x1_num + self.grad_num[0] * self.alpha
            x2_sym = self.x2_num + self.grad_num[1] * self.alpha
            # Создание переменной опять же, чтобы не потерять форумулы
            # grad1[0] = 6*x1_1 - 6*x2_1 + 8 = 6*(x1_0 + alpha*grad[0]) - 6*(x2_0 + alpha*grad[1]) + 8
            # grad1[1] = 10*x2_1 - 6*x1_1 + 9 = 10*(x2_0 + alpha*grad[0]) - 6*(x1_0 + alpha*grad[0]) + 9
            # grad1 - grad_sym - формула градиента
            # grad - grad_num - численное значение градиента предыдущей итерации
            grad_sym[0] = (self.grad_sym[0].subs({self.x1_sym: x1_sym, self.x2_sym: x2_sym}))
            grad_sym[1] = self.grad_sym[1].subs({self.x1_sym: x1_sym, self.x2_sym: x2_sym})
            # На данном шаге просто считаем альфу, как grad*grad1=0
            alpha = solve(self.grad_num[0] * grad_sym[0] + self.grad_num[1] * grad_sym[1])[0]
            # Подставляем численное значение альфы
            self.x1_num = float(self.x1_num + alpha * self.grad_num[0])
            self.x2_num = float(self.x2_num + alpha * self.grad_num[1])
            # Подставляем численное значение х1_1, х2_1
            self.grad_num = [float(self.grad_sym[0].evalf(subs={self.x1_sym: self.x1_num,
                                                          self.x2_sym: self.x2_num})),
                             float(self.grad_sym[1].evalf(subs={self.x1_sym: self.x1_num,
                                                          self.x2_sym: self.x2_num}))]
            # Считаем норму матрицы
            self.norm = np.sqrt(float(self.grad_num[0] ** 2) + float(self.grad_num[1] ** 2))
            # Считываем данные
            self.__collect_data()
            self.it += 1

    def __collect_data(self):
        self.data['x1'].append(self.x1_num)
        self.data['x2'].append(self.x2_num)
        self.data['Градиент'].append(self.grad_num)
        self.data['Итерация'].append(self.it)
        self.data['Норма матрицы'].append(self.norm)
        self.data['Альфа'].append(self.norm)

    def print(self):
        data = pd.DataFrame(self.data)
        print(data)

    def plot(self, is_anim=False, is_3d=False):
        # Создаю лямбда-функцию из выражения Sympy
        func = lambdify((self.x1_sym, self.x2_sym), self.func)
        # Создание рамок
        fr = Frame(self.data['x1'], self.data['x2'])
        x1 = np.arange(fr.left, fr.right, fr.step_w)
        x2 = np.arange(fr.down, fr.up, fr.step_h)
        x1, x2 = np.meshgrid(x1, x2)
        z = func(x1, x2)

        # Создание фигуры для рисования графика
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('$x1$')
        ax.set_ylabel('$x2$')
        ax.set_zlabel('$z$')

        # Рисование поверхности
        if is_3d:
            ax.plot_surface(x1, x2, z, cmap=mpl.cm.coolwarm)
        else:
            ax.view_init(90, 0)

        # Рисование плоскости уровня на плоскости хОy
        cset = ax.contourf(x1, x2, z, zdir='z', offset=z.min(), cmap=mpl.cm.coolwarm)

        # Рисование направления градиента
        x1_c = np.array(self.data['x1'])
        x2_c = np.array(self.data['x2'])
        z_c = func(x1_c, x2_c)
        if is_anim:
            anim = GraphAnimation(x1_c, x2_c, func, fig, ax)
        else:
            ax.plot(x1_c, x2_c, z_c, label='Направление градиента')

        ax.legend()

        plt.show()


a = GradientDescent(1, 3, '3*x1**2+5*x2**2-6*x1*x2+8*x1+9*x2', tol=0.001)
a.gradient_descent()
a.plot()