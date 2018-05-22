function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X * theta;
    error =  (h - y);
    Prod = X' * error; 
    %'  to workaround formatting in Visual Studio Code

    % Dimensions: dim(X) = m x n;   dim(theta) = n x 1; dim(h) = m x 1; dim(y) = m x 1
    % Note: In order to calculate Prod, we have to transpose X because dim(X) = m x n, whereas dim(error) = m x 1
    % After transpongs X, dim(X transpose) = n x m

    newTheta = (alpha / m ) * Prod;
    theta = theta - newTheta;

    % ============================================================

    % Save the cost J in every iteration   
    J_history(iter) = computeCost(X, y, theta);
end

fprintf("GD>>>J_history final = %f\n", J_history(num_iters));

end
