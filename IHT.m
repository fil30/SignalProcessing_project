% Function used for Iterative Hard Thresholding algorithm

function [s, error_vector, count] = IHT(y, theta, K)

% theta matrix dimensions
M = size(theta, 1);
N = size(theta, 2);

s = zeros(N, 1);         % Initialize the sparse vector to be estimated
u = 0.5;                 % Influence coefficient used for updating s
error_vector = [];       % Holds the error iteration by iteration
count = 0;               % Keep track of the number of iterations

for i = 1:K

    count = count + 1;    
    s_new = s + u * theta' * (y - theta * s);

    % Sorting in descending order to get the largest values
    [~, index] = sort(abs(s_new), 'descend'); 

    % Mantain only the largest values
    s_new(index(K+1:end)) = 0;

    error = norm(y - theta * s_new);   % Calculate the error
    error_vector(i) = error;           % Store the error in the vector

    s = s_new;                         % Update the s value

    % If the error is small enough exit the loop
    if error_vector(i) < 1e-6
        break;      
    end

end
end


