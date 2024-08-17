"""
- mean square error, cost function, gradient descent and learning rate. these are some of the important concept in ML.

area = [2600, 3000, 3200, 3600, 4000]
price = [550k, 565k, 610k, 680k, 725k]

y = mx + b
price = 135.78 * area + 180616.43

- our goal is to derive this equation
- that equation is nothing but blue line we plot in linear regression which is the best fit line going through all
  these data points
- right now these data points are scattered, so it's not possible to draw the perfect line, but you draw line which is
  kind of best fit.
- but the problem here is we might have many lines that can potentially go through these data points
- my data set is very simple here if we have heavy dataset that data points would be scattered all over the place
  then drawing these lines becomes even more difficult
- how do we know which of these lines is the best fit line, so that's the problem we are having

- we can get the best line by finding m and b using gradient descent.
- gradient descent is an algorithm that finds best fit line for given training data set

"""

import numpy as np


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
