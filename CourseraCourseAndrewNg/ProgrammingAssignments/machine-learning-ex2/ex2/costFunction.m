function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% size(X) = m x n; size(theta) = n x 1 (3 X 1). So size(hX) = m x 1 = 100 x 1
hX = sigmoid(X * theta);
part1 = y .* log(hX);
part2 = (ones(m, 1) - y) .* log(1 - hX);

J = (-1 / m) * sum(part1 + part2);

% to find the gradient, refer the formula in the assigment PDF
% since hX - y is 100 x 1 and X is 100 x n, we have to transpose X to multiply by hX - y
% we want to do grad[0] = (x0[1] * (hX-y)[1]) + (x0[2] * (hX-y)[2]) + .. (x0[100] * (hX-y)[100]) 
%     and then divide the whole thing by m
% similary calculate grad[1], grad[2] etc...
% So we have to do regular matrix multiplication instead of element-by-element mult
% Note: do not put sum in the forumla becuase matrix multiplication will do the sum
grad = (1 / m) * (X' * (hX - y));

% =============================================================

end
