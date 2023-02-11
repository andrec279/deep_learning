import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def nonlinear_forward(self, x, func):
        if func == 'relu':
            return torch.maximum(torch.zeros_like(x), x)
        elif func == 'sigmoid':
            return 1 / (1 + torch.exp(-x))
        elif func == 'identity':
            return x
        else: raise ValueError('invalid choice of func')

    def nonlinear_backward(self, x, func):
        grad = torch.ones_like(x)
        if func == 'relu':
            grad = grad * (x > 0).float()
        elif func == 'identity':
            grad = grad
        elif func == 'sigmoid':
            grad = grad * ((torch.exp(-x)) / (1 + torch.exp(-x))**2)
        else: raise ValueError('invalid choice of func')

        return grad

    
    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.x = x
        self.s1 = x @ self.parameters['W1'].T + self.parameters['b1'] # (batch_size, linear1_out)
        self.a1 = self.nonlinear_forward(self.s1, self.f_function) # (batch_size, linear1_out)
        self.s2 = self.a1 @ self.parameters['W2'].T + self.parameters['b2'] # (batch_size, linear2_out)
        self.y_hat = self.nonlinear_forward(self.s2, self.g_function) # (batch_size, linear2_out)

        return self.y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function

        self.grads['dJdW2'] = (dJdy_hat * self.nonlinear_backward(self.s2, self.g_function)).T @ self.a1
        self.grads['dJdb2'] = torch.sum((dJdy_hat * self.nonlinear_backward(self.s2, self.g_function)).T, 1)
        self.grads['dJdW1'] = (dJdy_hat * self.nonlinear_backward(self.s2, self.g_function) @ self.parameters['W2'] \
                    * self.nonlinear_backward(self.s1, self.f_function)).T @ self.x
        self.grads['dJdb1'] = torch.sum((dJdy_hat * self.nonlinear_backward(self.s2, self.g_function) @ self.parameters['W2'] \
                    * self.nonlinear_backward(self.s1, self.f_function)).T, 1)

        # for param in self.parameters:
        #     grad_key = 'dJd' + param
        #     self.parameters[param] -= self.grads[grad_key]

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    J = (torch.norm(y_hat - y) ** 2) / (y.size(0)*y.size(1))
    dJdy_hat = (2 * (y_hat - y)) / (y.size(0)*y.size(1))

    return J, dJdy_hat


    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    K = y.size(1)
    n = y.size(0)
    
    J = (1 / (K*n)) * torch.sum(-(y*torch.log(y_hat) + (1 - y)*torch.log(1-y_hat)))
    dJdy_hat = (1 / (K*n)) * (- y/y_hat - (y-1)/(1-y_hat))

    return J, dJdy_hat











