function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% size(X) = m x n; size(theta) = n x 1 (3 X 1). So size(hX) = m x 1 = 100 x 1
hX = sigmoid(X * theta);
part1 = y .* log(hX);
part2 = (ones(m, 1) - y) .* log(1 - hX);
n = size(theta);

% Remember: Do not regularize theta1. So it should be excluded from J
% good - part3 = (lambda / (2 * m)) * sum(theta(2:n).^2);
part3 = (lambda * sum(theta(2:n) .^ 2)) / (2 * m);

J = (-sum(part1 + part2) / m) + part3;

% to find the gradient, refer the formula in the assigment PDF
% since hX - y is 100 x 1 and X is 100 x n, we have to transpose X to multiply by hX - y
% we want to do grad[0] = (x0[1] * (hX-y)[1]) + (x0[2] * (hX-y)[2]) + .. (x0[100] * (hX-y)[100]) 
%     and then divide the whole thing by m
% similary calculate grad[1], grad[2] etc...
% So we have to do regular matrix multiplication instead of element-by-element mult
% Note: do not put sum in the forumla becuase matrix multiplication will do the sum

% vector form
% grad(1) = (1 / m) * (X(:,1)' * (hX - y));  % for j = 0
% grad(2:n) = ((1 / m) * (X(:, 2:n)' * (hX - y))) + ((lambda / m) * theta(2:n));  % for j >= 1

for j = 1:n
 if (j == 1)
  grad(j) = (1 / m) * (X(:,1)' * (hX - y));
 else
  grad(j) = ((1 / m) * (X(:,j)' * (hX - y))) + ((lambda / m) * theta(j));
 endif
end

% =============================================================

end
