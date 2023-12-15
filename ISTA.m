% Function used for Iterative Soft Thresholding algorithm

function [s, error_vector, count] = ISTA(y, theta, K)

% theta matrix dimensions
M = size(theta, 1); 
N = size(theta, 2);

t = 1 / norm(theta) ^ 2;   % Assign a value to the t step (1 / norm(theta) ^ 2 is a common value)
lambda = 0.01;             % Assign a value to the regularization term

sk_1 = zeros(N,1);    % Initialize the sparse vector to be estimated
error_vector = [];    % Holds the error iteration by iteration
count = 0;            % Keep track of the number of iterations

for i = 1:K
    count = count + 1;
    grad = 2 * theta' * (theta * sk_1 - y);            % Calculate the gradient
    sk = sk_1 - t * grad;                              % Update sk value
    st = (max(abs(sk) - t * lambda, 0)) .* sign(sk);   % Define soft thresholding
    error_vector(i) = norm(theta * st - y);            % Calculate the error

    sk_1 = st;
    % If the error is small enough exit the loop
    if error_vector(i) < 1e-6                      
         break;
    end
end
s = sk_1;
end



