% Function used for Orthogonal Matching Pursuit algorithm

function [s, error_vector, count] = OMP(y, theta, K)

% theta matrix dimensions
M = size(theta, 1); 
N = size(theta, 2);

s = zeros(N,1);       % Initialize the sparse vector to be estimated   
s0 = [];              % Mantain the columns of theta useful for reconstruction
el = y;               % Residual value updated every iteration
count = 0;            % Keep track of the number of iterations
error_vector = [];    % Holds the error iteration by iteration
pos = [];             % Mantain the indexes of the columns of the theta matrix 

for i = 1:K
    count = count + 1;
    for j = 1:N                                 
        value(j) = abs(theta(:, j)' * el);         % Calculate all the products 
    end

    [~, index] = max(value);                       % Retrieve only the maximum value   
    s0 = [s0, theta(:,index)];                     % Update the vector s0 with the new column
    theta(:, index) = zeros(M,1);                  % Set this column of theta to 0

    ls = inv(s0' * s0) * s0' * y;                  % Resolve the Least squares problem on the defined suport
    el = y - s0 * ls;                              % Update residual for the next iteration
    error_vector(i) = norm(el);                    % Mantain the error of the iterations
    pos(i) = index;                                % Mantain the index of the column of theta choosed

    if error_vector(i) < 1e-6                      % If the error is small enough exit the loop
         break;
    end
end
s(pos) = ls;                                       % Reconstructed vector
end





