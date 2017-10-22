import matplotlib.pyplot as plt
from sympy import Symbol, solve
import numpy as np
import matplotlib.animation as animation
import pandas as pd
from matplotlib import cm

"""fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)
plt.show()"""


class Animation:
    def __init__(self, x_list, y_list):
        self.iter = 0
        self.fig, self.ax = plt.subplots()
        self.ax.grid(True)
        self.x_list = x_list
        self.y_list = y_list
        self.line, = self.ax.plot(self.x_list, self.y_list, color='green')
        xmin, xmax, step = np.array(x_list).min()-0.5, np.array(x_list).max()+0.5, 0.01
        ymin, ymax, step = np.array(y_list).min()-0.5, np.array(y_list).max()+0.5, 0.01
        x, y = np.meshgrid(np.arange(xmin, xmax + step, step), np.arange(ymin, ymax + step, step))
        #Cюда вставляй коэффиценты функции, x1 - x, x2 - y
        func = lambda x, y: 3 * x ** 2 + 5 * y ** 2 - 6 * x * y + 8 * x + 9 * y
        z = func(x, y)
        plt.pcolormesh(x, y, z, cmap=cm.coolwarm)
        plt.colorbar()
        plt.contour(x, y, z, colors='black')

    def init(self):
        self.line.set_ydata(np.ma.array(self.x_list, mask=True))
        return self.line,

    def animate(self, i):
        self.line.set_data(self.x_list[:i], self.y_list[:i])
        return self.line,

    def main_func(self):
        print(self.x_list)
        ani = animation.FuncAnimation(self.fig, self.animate, np.arange(1, 200), interval=1000, repeat_delay=0)
        ani.save('Градиентный спуск_м.mp4')
        plt.show()

#grad_func1 - сюда вставляй коэффиценты градиент1
#grad_func2 - сюда вставляй коэффиценты градиент2
f1 = lambda x1, x2: 6 *x1 - 6 * x2 + 8
f2 = lambda x1, x2: 10 * x2 - 6 * x1 + 9
#f1_1 = sympify('6*x1-6*x2+8')


def printing(df):
    df = pd.DataFrame(df, columns=['Итерация', 'Норма матрицы', 'Градиент', 'х1', 'x2', 'alpha'])
    #print(df.round(2))
    df.round(3).to_csv('Градзіентны сход_М.csv')


def gradient_descent(x1, x2, eps, m):
    #fig = plt.plot()
    k = 0
    x1_num = x1
    x2_num = x2
    #plt.plot(x1, x2)
    grad = (f1(x1_num,x2_num), f2(x1_num, x2_num))
    print(grad)
    err = np.sqrt(float(grad[0]**2) + float(grad[1]**2))
    x_list = [x1]
    y_list = [x2]
    df = []
    df.append([k, err, grad, x1, x2, 0] )
    while k < m and err > eps:
        alpha = Symbol('a')
        x1_sym = x1_num + grad[0]*alpha
        #print(x1_sym)
        x2_sym = x2_num + grad[1]*alpha
        #print(x2_sym)
        #grad_func1 - сюда вставляй коэффиценты градиент1
        #grad_func2 - сюда вставляй коэффиценты градиент2
        grad_func1 = 6 * x1_sym - 6 * x2_sym + 8
        grad_func2 = 10 * x2_sym - 6 * x1_sym + 9
        alpha = solve(grad[0]*grad_func1+grad[1]*grad_func2)
        x1_num = float(x1_num + alpha[0]*grad[0])
        x2_num = float(x2_num + alpha[0]*grad[1])
        grad = (f1(x1_num,x2_num), f2(x1_num, x2_num))
        err = np.sqrt(float(grad[0]**2) + float(grad[1]**2))
        x_list.append(x1_num)
        y_list.append(x2_num)
        df.append([k+1, err, grad, x1_num, x2_num, alpha[0]])
        #print('iteration = ', k+1)
        #print('gradient = {0:4f} {1:4f}'.format(grad[0], grad[1]))
        #print('modul_gradient = %.2f' % err)
        #print('x1 = %.2f' %  x1_num, ' x2 = %.2f' % x2_num)
        #print('alpha = %.2f' % alpha[0])
        #print(30*'=')
        #print(30*'=')
        k += 1
    printing(df)
    a = Animation(x_list, y_list)
    a.main_func()


gradient_descent(1, 3, 0.1, 100)

