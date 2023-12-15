close all
clc
clear
warning off

N = 2000;                      % Dimension of the sparse vector
K = 600;                       % Sparsity level
M = 2 * ceil(K * log(N / K));  % Number of observations

rng(10);  % Set seed for reproducibility

index = randperm(N);  % Define a random permutation of indices from 1 to N
x_orig = zeros(N, 1);

rng(11);

x_orig(index(1:K)) = randn(K, 1);   % Define a k-sparse representation

t = 0:N-1;

% Plot the original sparse vector
figure;
plot(t, x_orig);
title('Original signal representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

psi = eye(N);           % Define the basis in which the signal is sparse
phi = randn(M,N) / sqrt(M);       % Define the measurement matrix with variance 1 / M
phi = orth(phi')';      % Ortogonalize the measurement matrix
theta = phi * psi;      % Define the theta matrix
y = phi * x_orig;       % Simulate measurements

% Apply the ISTA function
tic
[s_r_ista, error_ista, count_ista] = ISTA(y, theta, K);
elapsed_time_ista = toc;
fprintf('ISTA algorithm completed in %.4f seconds.\n', elapsed_time_ista);
fprintf('ISTA algorithm completed in %d iterations. \n\n', count_ista);

% Apply the IHT function
tic
[s_r_iht, error_iht, count_iht] = IHT(y, theta, K);
elapsed_time_iht = toc;
fprintf('IHT algorithm completed in %.4f seconds.\n', elapsed_time_iht);
fprintf('IHT algorithm completed in %d iterations. \n\n', count_iht);

% Apply OMP function
tic
[s_r_omp, error_omp, count_omp] = OMP(y,theta, K);
elapsed_time_omp = toc;
fprintf('OMP algorithm completed in %.4f seconds.\n', elapsed_time_omp);
fprintf('OMP algorithm completed in %d iterations. \n\n', count_omp);

% Reconstruct the original signal
x_r_iht = psi * s_r_iht;
x_r_omp = psi * s_r_omp;
x_r_ista = psi * s_r_ista;

% Plot the original signal and the reconstructed signal by ISTA algorithm
figure;

subplot(2,1,1)
plot(t, x_orig, 'r');  % Original signal
title('Original signal representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

subplot(2,1,2)
plot(t, x_r_ista, 'r');  % Reconstructed signal
title('ISTA sparse representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');


res_ista = norm(x_r_ista - x_orig);
fprintf('Residual ISTA：%d\n', res_ista);

% Plot the original signal and the reconstructed signal by IHT algorithm
figure;

subplot(2,1,1)
plot(t, x_orig, 'r');  % Original signal
title('Original signal representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

subplot(2,1,2)
plot(t, x_r_iht, 'r');  % Reconstructed signal
title('IHT sparse representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

res_iht = norm(x_r_iht - x_orig);
fprintf('Residual IHT：%d\n', res_iht);

% Plot the original signal and the reconstructed signal by OMP algorithm
figure;

subplot(2,1,1)
plot(t, x_orig, 'r');  % Original signal
title('Original signal representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

subplot(2,1,2)
plot(t, x_r_omp, 'r');  % Reconstructed signal
title('OMP sparse representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

res_omp = norm(x_r_omp - x_orig);
fprintf('Residual OMP：%d\n', res_omp);

% Plot all the signals together
figure;

subplot(4,1,1);
plot(t, x_orig);
title('Original signal representation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

subplot(4,1,2);
plot(t, x_r_ista);
title('ISTA sparse approximation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

subplot(4,1,3);
plot(t, x_r_iht);
title('IHT sparse approximation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

subplot(4,1,4);
plot(t, x_r_omp);
title('OMP sparse approximation');
xlabel('Discrete Time (t)');
ylabel('Amplitude');

figure;

% Plot ISTA error
subplot(3,1,1);
plot(1:length(error_ista), error_ista, 'r', 'LineWidth', 2);
title('ISTA error');
xlabel('Number of iterations');
ylabel('Error');

% Plot IHT error
subplot(3,1,2);
plot(1:length(error_iht), error_iht, 'r', 'LineWidth', 2);
title('IHT error');
xlabel('Number of iterations');
ylabel('Error');

% Plot OMP error
subplot(3,1,3);
plot(1:length(error_omp), error_omp, 'r', 'LineWidth', 2);
title('OMP error');
xlabel('Number of iterations');
ylabel('Error');